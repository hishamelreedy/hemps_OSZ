#include "niles_health.h"
#include "seek.h"
#include "../include/plasma.h"
#include "utils.h"

#ifndef NILES_BCAST_APP_PE_ONLY
#define NILES_BCAST_APP_PE_ONLY 0
#endif

#ifndef NILES_LINK_SCORE_MAX
#define NILES_LINK_SCORE_MAX 255
#endif

#if NILES_BCAST_APP_PE_ONLY
static const unsigned int NILES_APP_PE_LIST[] = {0x0000, 0x0100, 0x0200, 0x0300};
#define NILES_APP_PE_COUNT 4
#else
#define NILES_APP_PE_COUNT (NILES_MESH_X * NILES_MESH_Y)
#endif

#define NILES_DIR_EAST  0
#define NILES_DIR_WEST  1
#define NILES_DIR_NORTH 2
#define NILES_DIR_SOUTH 3
/* 4x4 mesh: max Manhattan hops = 6; pack all dirs in 16 bits. */
#define NILES_PATH_MAX_HOPS 6

static unsigned char niles_fault_table[NILES_FAULT_TABLE_SIZE];
static unsigned char niles_link_table[4][NILES_FAULT_TABLE_SIZE];
static unsigned int niles_bcast_sent;
static unsigned int niles_bcast_received;
static unsigned int niles_hb_sent;
static unsigned int niles_hb_received;
static unsigned int niles_fault_marks;
static unsigned int niles_hb_seq;
static unsigned int niles_obs_path_enc;
static unsigned int niles_obs_source_pe;
static int niles_obs_producer;
static int niles_obs_valid;

static unsigned int niles_peer_count(void) {
    return (unsigned int)NILES_APP_PE_COUNT;
}

static void niles_foreach_peer(unsigned int sender_pe,
                               void (*visit)(unsigned int target, void *ctx),
                               void *ctx) {
#if NILES_BCAST_APP_PE_ONLY
    int i;
    for (i = 0; i < (int)NILES_APP_PE_COUNT; i++)
        visit(NILES_APP_PE_LIST[i], ctx);
#else
    int x, y;
    (void)sender_pe;
    for (x = 0; x < NILES_MESH_X; x++) {
        for (y = 0; y < NILES_MESH_Y; y++) {
            visit((unsigned int)((x << 8) | y), ctx);
        }
    }
#endif
}

static int niles_pe_index(unsigned int pe_addr) {
    int x = (pe_addr >> 8) & 0xF;
    int y = pe_addr & 0xF;
    if (x < 0 || x >= NILES_MESH_X || y < 0 || y >= NILES_MESH_Y)
        return -1;
    return x * NILES_MESH_Y + y;
}

static unsigned int niles_pe_expand(unsigned char compact) {
    return ((unsigned int)(compact >> 4) << 8) | (unsigned int)(compact & 0xF);
}

static unsigned char niles_pe_compact(unsigned int pe_addr) {
    return (unsigned char)(((pe_addr >> 8) & 0xF) << 4) | (unsigned char)(pe_addr & 0xF);
}

static int niles_write_xy_path(char *path_buffer, unsigned int source, unsigned int target) {
    int i = 0;
    int current_x = (source >> 8) & 0xF;
    int current_y = source & 0xF;
    int target_x = (target >> 8) & 0xF;
    int target_y = target & 0xF;

    while (current_x < target_x) {
        path_buffer[i++] = (char)EAST;
        current_x++;
    }
    while (current_x > target_x) {
        path_buffer[i++] = (char)WEST;
        current_x--;
    }
    while (current_y < target_y) {
        path_buffer[i++] = (char)NORTH;
        current_y++;
    }
    while (current_y > target_y) {
        path_buffer[i++] = (char)SOUTH;
        current_y--;
    }
    return i;
}

/*
 * Pack all hops (up to NILES_PATH_MAX_HOPS) into 16 bits:
 *   [15:12] length (0..6)
 *   [11:0]  dir_i in 2-bit fields (dir0 in [1:0], dir1 in [3:2], ...)
 */
static unsigned int niles_path_pack_u16(const char *path, int path_len) {
    unsigned int packed = 0;
    int i;
    int n = path_len;

    if (n < 0)
        n = 0;
    if (n > NILES_PATH_MAX_HOPS)
        n = NILES_PATH_MAX_HOPS;
    packed = ((unsigned int)n & 0xF) << 12;
    for (i = 0; i < n; i++)
        packed |= (((unsigned int)path[i] & 0x3) << (2 * i));
    return packed;
}

static int niles_path_unpack_u16(unsigned int packed, char *path) {
    int n = (int)((packed >> 12) & 0xF);
    int i;

    if (n > NILES_PATH_MAX_HOPS)
        n = NILES_PATH_MAX_HOPS;
    for (i = 0; i < n; i++)
        path[i] = (char)((packed >> (2 * i)) & 0x3);
    return n;
}

static int niles_build_observed_path(unsigned int suspect_pe, unsigned int reporter_pe,
                                     char *path) {
    if (niles_obs_valid)
        return niles_path_unpack_u16(niles_obs_path_enc, path);

    if (suspect_pe == reporter_pe)
        return 0;

    return niles_write_xy_path(path, suspect_pe, reporter_pe);
}

static void niles_link_score_inc(int dir, int x, int y) {
    int idx;

    if (x < 0 || x >= NILES_MESH_X || y < 0 || y >= NILES_MESH_Y)
        return;
    idx = x * NILES_MESH_Y + y;
    if (niles_link_table[dir][idx] < NILES_LINK_SCORE_MAX)
        niles_link_table[dir][idx]++;
}

static void niles_apply_path_scores(unsigned int start_pe, const char *path, int path_len) {
    int x = (start_pe >> 8) & 0xF;
    int y = start_pe & 0xF;
    int i;

    for (i = 0; i < path_len; i++) {
        niles_link_score_inc((int)path[i], x, y);
        switch (path[i]) {
        case EAST:  x++; break;
        case WEST:  x--; break;
        case NORTH: y++; break;
        case SOUTH: y--; break;
        default: break;
        }
        if (x < 0 || x >= NILES_MESH_X || y < 0 || y >= NILES_MESH_Y)
            break;
    }
}

static int niles_link_valid(int dir, int x, int y) {
    if (x < 0 || x >= NILES_MESH_X || y < 0 || y >= NILES_MESH_Y)
        return 0;
    if (dir == NILES_DIR_EAST && x >= NILES_MESH_X - 1)
        return 0;
    if (dir == NILES_DIR_WEST && x <= 0)
        return 0;
    if (dir == NILES_DIR_NORTH && y >= NILES_MESH_Y - 1)
        return 0;
    if (dir == NILES_DIR_SOUTH && y <= 0)
        return 0;
    return 1;
}

static void niles_print_dir_name(int dir) {
    switch (dir) {
    case NILES_DIR_EAST:  puts("EAST"); break;
    case NILES_DIR_WEST:  puts("WEST"); break;
    case NILES_DIR_NORTH: puts("NORTH"); break;
    default:              puts("SOUTH"); break;
    }
}

static void niles_print_path_news(const char *path, int path_len) {
    int i;

    if (path_len <= 0) {
        puts("-");
        return;
    }
    for (i = 0; i < path_len; i++) {
        if (i > 0) {
            puts(" ");
        }
        niles_print_dir_name((int)path[i]);
    }
}

static void niles_echo_path_enc(unsigned int enc, const char *path, int path_len) {
    puts("[NILES-PATH] enc=");
    puts(itoh(enc));
    puts(" hops=");
    puts(itoa(path_len));
    puts(" decoded=");
    niles_print_path_news(path, path_len);
    puts("\n");
}

static void niles_print_bcast_decode(const char *tag, unsigned int source, unsigned int payload,
                                     int path_len, const char *path) {
    unsigned int path_enc = (source >> 16) & 0xFFFF;
    unsigned char reporter_compact = (unsigned char)((source >> 8) & 0xFF);
    unsigned int reporter_pe = niles_pe_expand(reporter_compact);
    unsigned int tick_lo = source & 0xFF;
    unsigned char payload8 = (unsigned char)(payload & 0xFF);
    int layer_id = (payload8 >> 4) & 0xF;
    int suspect_x = (payload8 >> 2) & 0x3;
    int suspect_y = payload8 & 0x3;
    int suspect_pe = (suspect_x << 8) | suspect_y;
    unsigned int path_start = niles_pe_expand((unsigned char)((payload >> 8) & 0xFF));

    puts(tag);
    puts(" source=");
    puts(itoh(source));
    puts(" payload8=");
    puts(itoh((unsigned int)payload8));
    puts(" path_enc=");
    puts(itoh(path_enc));
    puts(" tick_lo=");
    puts(itoa((int)tick_lo));
    puts(" reporter_compact=");
    puts(itoa((int)reporter_compact));
    puts(" reporter_pe=");
    puts(itoa(reporter_pe));
    puts(" path_start=");
    puts(itoa((int)path_start));
    puts(" layer=");
    puts(itoa(layer_id));
    puts(" suspect_x=");
    puts(itoa(suspect_x));
    puts(" suspect_y=");
    puts(itoa(suspect_y));
    puts(" suspect_pe=");
    puts(itoa(suspect_pe));
    puts(" path_hops=");
    puts(itoa(path_len));
    puts(" path=");
    niles_print_path_news(path, path_len);
    puts("\n");
}

static void niles_link_table_print(unsigned int tick) {
    int dir, y, x;
    unsigned int self = get_net_address() & 0xFFFF;

    for (dir = 0; dir < 4; dir++) {
        puts("[NILES-LINK-TABLE] dir=");
        niles_print_dir_name(dir);
        puts(" self=");
        puts(itoa(self));
        puts(" tick=");
        puts(itoa(tick));
        puts("\n");

        for (y = 0; y < NILES_MESH_Y; y++) {
            puts("[NILES-LINK-TABLE] y=");
            puts(itoa(y));
            for (x = 0; x < NILES_MESH_X; x++) {
                int idx = x * NILES_MESH_Y + y;
                puts(" ");
                if (niles_link_valid(dir, x, y)) {
                    puts(itoa((int)niles_link_table[dir][idx]));
                } else {
                    puts("-");
                }
            }
            puts("\n");
        }
    }
}

static void niles_fault_table_print_matrix(unsigned int tick) {
    int y, x;
    unsigned int self = get_net_address() & 0xFFFF;

    puts("[NILES-VISION-MAP] self=");
    puts(itoa(self));
    puts(" tick=");
    puts(itoa(tick));
    puts(" marks=");
    puts(itoa(niles_fault_marks));
    puts(" size=");
    puts(itoa(NILES_MESH_X));
    puts("x");
    puts(itoa(NILES_MESH_Y));
    puts("\n");

    for (y = 0; y < NILES_MESH_Y; y++) {
        puts("[NILES-VISION-MAP] y=");
        puts(itoa(y));
        for (x = 0; x < NILES_MESH_X; x++) {
            int idx = x * NILES_MESH_Y + y;
            puts(" ");
            puts(itoa((int)niles_fault_table[idx]));
        }
        puts("\n");
    }
}

static void niles_print_all_tables(unsigned int tick) {
    niles_link_table_print(tick);
    niles_fault_table_print_matrix(tick);
}

struct niles_seek_ctx {
    unsigned int service;
    unsigned int source;
    unsigned int payload;
    int first;
};

static void niles_seek_pace(void) {
    unsigned int deadline = MemoryRead(TICK_COUNTER) + 20000u;
    while (MemoryRead(TICK_COUNTER) < deadline) {
    }
}

static void niles_seek_to_peer(unsigned int target, void *ctx) {
    struct niles_seek_ctx *s = (struct niles_seek_ctx *)ctx;
    if (!s->first)
        niles_seek_pace();
    s->first = 0;
    Seek(s->service, s->source, target, s->payload);
}

void niles_fault_table_init(void) {
    int i, d;

    for (i = 0; i < NILES_FAULT_TABLE_SIZE; i++)
        niles_fault_table[i] = 0;
    for (d = 0; d < 4; d++) {
        for (i = 0; i < NILES_FAULT_TABLE_SIZE; i++)
            niles_link_table[d][i] = 0;
    }
    niles_bcast_sent = 0;
    niles_bcast_received = 0;
    niles_hb_sent = 0;
    niles_hb_received = 0;
    niles_fault_marks = 0;
    niles_hb_seq = 0;
    niles_obs_path_enc = 0;
    niles_obs_source_pe = 0;
    niles_obs_producer = 0;
    niles_obs_valid = 0;
}

void niles_note_rx_path(unsigned int path_enc, unsigned int source_pe, int producer_task) {
    char path[8];
    int path_len;

    path_enc &= 0xFFFF;
    path_len = niles_path_unpack_u16(path_enc, path);
    niles_obs_path_enc = path_enc;
    niles_obs_source_pe = source_pe & 0xFFFF;
    niles_obs_producer = producer_task;
    niles_obs_valid = (path_len > 0);

    puts("[NILES-PATH-RX] producer=");
    puts(itoa(producer_task));
    puts(" src_pe=");
    puts(itoa((int)niles_obs_source_pe));
    puts(" enc=");
    puts(itoh(path_enc));
    puts(" hops=");
    puts(itoa(path_len));
    puts(" decoded=");
    niles_print_path_news(path, path_len);
    puts("\n");
}

void niles_fault_mark(unsigned int pe_addr, int layer_id, unsigned int tick) {
    int idx = niles_pe_index(pe_addr);
    if (idx < 0)
        return;
    if (niles_fault_table[idx] == 0)
        niles_fault_marks++;
    niles_fault_table[idx] = (unsigned char)(layer_id + 1);
    puts("[NILES-FAULT-TABLE] pe=");
    puts(itoa(pe_addr));
    puts(" layer=");
    puts(itoa(layer_id));
    puts(" tick=");
    puts(itoa(tick));
    puts("\n");
}

int niles_fault_query(unsigned int pe_addr) {
    int idx = niles_pe_index(pe_addr);
    if (idx < 0)
        return 0;
    return (int)niles_fault_table[idx];
}

/*
 * Seek encoding:
 *   source [31:16] path_enc (len+dirs), [15:8] reporter compact, [7:0] tick_lo
 *   payload [15:8] path_start PE, [7:4] layer, [3:2] suspect_x, [1:0] suspect_y
 */
void niles_broadcast_ic_fault(int suspect_pe, int layer_id, unsigned int reporter_pe) {
#if NILES_ENABLE_BROADCAST
    char path[8];
    int path_len;
    unsigned int path_enc;
    unsigned int path_start;
    unsigned int tick = MemoryRead(TICK_COUNTER);
    unsigned int source;
    unsigned int payload;
    struct niles_seek_ctx ctx;

    path_len = niles_build_observed_path((unsigned int)suspect_pe, reporter_pe, path);
    if (niles_obs_valid)
        path_enc = niles_obs_path_enc;
    else
        path_enc = niles_path_pack_u16(path, path_len);
    path_start = niles_obs_valid ? niles_obs_source_pe : (unsigned int)suspect_pe;
    niles_echo_path_enc(path_enc, path, path_len);

    niles_fault_mark((unsigned int)suspect_pe, layer_id, tick);
    niles_apply_path_scores(path_start, path, path_len);

    /* source: [31:16] full path enc, [15:8] reporter, [7:0] tick_lo */
    source = (path_enc << 16)
           | ((unsigned int)niles_pe_compact(reporter_pe) << 8)
           | (tick & 0xFF);
    payload = ((unsigned int)niles_pe_compact(path_start) << 8)
            | ((layer_id & 0xF) << 4)
            | (((suspect_pe >> 8) & 0x3) << 2)
            | (suspect_pe & 0x3);

    ctx.service = NILES_IC_FAULT_BROADCAST;
    ctx.source = source;
    ctx.payload = payload;
    ctx.first = 1;
    niles_foreach_peer(reporter_pe, niles_seek_to_peer, &ctx);
    niles_bcast_sent += niles_peer_count();

    puts("[NILES-BCAST] suspect_pe=");
    puts(itoa(suspect_pe));
    puts(" layer=");
    puts(itoa(layer_id));
    puts(" reporter=");
    puts(itoa(reporter_pe));
    puts(" path_start=");
    puts(itoa((int)path_start));
    puts(" path_hops=");
    puts(itoa(path_len));
    puts(" path=");
    niles_print_path_news(path, path_len);
    puts(" peers=");
    puts(itoa(niles_peer_count()));
    puts("\n");

    niles_print_bcast_decode("[NILES-BCAST-DEC]", source, payload, path_len, path);

    niles_print_all_tables(tick);
#else
    (void)suspect_pe;
    (void)layer_id;
    (void)reporter_pe;
#endif
}

void niles_handle_ic_fault_broadcast(unsigned int source, unsigned int target, unsigned int payload) {
    unsigned int path_enc = (source >> 16) & 0xFFFF;
    unsigned int reporter_pe = niles_pe_expand((unsigned char)((source >> 8) & 0xFF));
    unsigned int path_start = niles_pe_expand((unsigned char)((payload >> 8) & 0xFF));
    int layer_id = (payload >> 4) & 0xF;
    int suspect_pe = (((payload >> 2) & 0x3) << 8) | (payload & 0x3);
    unsigned int tick = MemoryRead(TICK_COUNTER);
    char path[8];
    int path_len;

    (void)target;

    path_len = niles_path_unpack_u16(path_enc, path);

    niles_bcast_received++;
    puts("[NILES-BCAST-RCV] reporter=");
    puts(itoa(reporter_pe));
    puts(" suspect_pe=");
    puts(itoa(suspect_pe));
    puts(" layer=");
    puts(itoa(layer_id));
    puts(" path_start=");
    puts(itoa((int)path_start));
    puts(" path_hops=");
    puts(itoa(path_len));
    puts(" path=");
    niles_print_path_news(path, path_len);
    puts(" tick=");
    puts(itoa(tick));
    puts("\n");

    niles_echo_path_enc(path_enc, path, path_len);
    niles_print_bcast_decode("[NILES-BCAST-RCV-DEC]", source, payload, path_len, path);

    niles_fault_mark((unsigned int)suspect_pe, layer_id, tick);
    niles_apply_path_scores(path_start, path, path_len);
    niles_print_all_tables(tick);
}

struct niles_hb_ctx {
    unsigned int sender_pe;
    unsigned int source;
    unsigned int payload;
    unsigned int sent;
};

static void niles_hb_to_peer(unsigned int target, void *ctx) {
    struct niles_hb_ctx *h = (struct niles_hb_ctx *)ctx;
    if (target == (h->sender_pe & 0xFFFF))
        return;
    if (h->sent > 0)
        niles_seek_pace();
    Seek(NILES_HEARTBEAT, h->source, target, h->payload);
    h->sent++;
}

void niles_send_heartbeat(unsigned int sender_pe) {
#if NILES_ENABLE_HEARTBEAT
    unsigned int tick = MemoryRead(TICK_COUNTER);
    unsigned int source = (tick << 16) | (sender_pe & 0xFFFF);
    unsigned int golden_sig = 0x4E494C45u ^ niles_hb_seq;
    unsigned int payload = (golden_sig << 16) | (niles_hb_seq & 0xFFFF);
    struct niles_hb_ctx ctx;

    niles_hb_seq++;
    ctx.sender_pe = sender_pe;
    ctx.source = source;
    ctx.payload = payload;
    ctx.sent = 0;
    niles_foreach_peer(sender_pe, niles_hb_to_peer, &ctx);
    niles_hb_sent++;

    puts("[NILES-HB] sender=");
    puts(itoa(sender_pe));
    puts(" seq=");
    puts(itoa(niles_hb_seq));
    puts(" tick=");
    puts(itoa(tick));
    puts("\n");
#else
    (void)sender_pe;
#endif
}

void niles_handle_heartbeat(unsigned int source, unsigned int target, unsigned int payload) {
    unsigned int sender = source & 0xFFFF;
    unsigned int tick = source >> 16;
    (void)target;
    (void)payload;
    niles_hb_received++;
    puts("[NILES-HB-RCV] from=");
    puts(itoa(sender));
    puts(" tick=");
    puts(itoa(tick));
    puts("\n");
}

void niles_hb_wait_interval(void) {
    unsigned int start = MemoryRead(TICK_COUNTER);
    unsigned int deadline = start + (unsigned int)NILES_HB_INTERVAL_US * 100u;
    while (MemoryRead(TICK_COUNTER) < deadline) {
    }
}

unsigned int niles_stats_bcast_sent(void) { return niles_bcast_sent; }
unsigned int niles_stats_bcast_received(void) { return niles_bcast_received; }
unsigned int niles_stats_hb_sent(void) { return niles_hb_sent; }
unsigned int niles_stats_hb_received(void) { return niles_hb_received; }
unsigned int niles_stats_fault_marks(void) { return niles_fault_marks; }
