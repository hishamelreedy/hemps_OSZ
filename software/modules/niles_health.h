#ifndef NILES_HEALTH_H_
#define NILES_HEALTH_H_

#include "../include/mlp784_niles_config.h"

/*
 * NILES-on-HEMPS: distributed fault table + IC alert broadcast + golden heartbeat.
 *
 * Logs (grep in sim logs):
 *   [NILES-PATH]      — IC-alert path enc= (16-bit, all hops ≤6) + decoded EAST/WEST/...
 *   [NILES-PATH-RX]   — dest Echo of CNN msg[1] observed path (after MAGIC)
 *   [NILES-BCAST]     — IC alert fan-out; full path scores health maps
 *   [NILES-BCAST-DEC] — decoded Seek source/payload on send
 *   [NILES-BCAST-RCV] — IC alert received (mesh PEs)
 *   [NILES-BCAST-RCV-DEC] — decoded on receive
 *   [NILES-FAULT-TABLE] / [NILES-LINK-TABLE] / [NILES-VISION-MAP]
 *   [NILES-PSB] / [NILES-HB-SEM] / [NILES-HB] / [NILES-HB-RCV]
 *
 * Path: CNN msg[1] after MAGIC. SwitchControl grant writes the hop in
 *   Hermes_buffer (before the port HT). Dest Echo is [NILES-PATH-RX].
 *   (XY fallback only if pad is empty).
 * Seek source: [31:16] path_enc, [15:8] reporter, [7:0] tick_lo.
 * Seek payload: [15:8] path_start PE, [7:4] layer, [3:2] suspect_x, [1:0] suspect_y.
 */

#ifndef NILES_MESH_X
#define NILES_MESH_X 4
#endif

#ifndef NILES_MESH_Y
#define NILES_MESH_Y 4
#endif

#ifndef NILES_ENABLE_BROADCAST
#define NILES_ENABLE_BROADCAST 1
#endif

#ifndef NILES_ENABLE_HEARTBEAT
#define NILES_ENABLE_HEARTBEAT 1
#endif

#ifndef NILES_HB_INTERVAL_US
#define NILES_HB_INTERVAL_US 1000
#endif

#ifndef NILES_HB_COUNT
#define NILES_HB_COUNT 5
#endif

#ifndef NILES_BCAST_APP_PE_ONLY
#define NILES_BCAST_APP_PE_ONLY 0
#endif

#define NILES_FAULT_TABLE_SIZE (NILES_MESH_X * NILES_MESH_Y)

void niles_fault_table_init(void);
void niles_fault_mark(unsigned int pe_addr, int layer_id, unsigned int tick);
int niles_fault_query(unsigned int pe_addr);

void niles_note_rx_path(unsigned int path_enc, unsigned int source_pe, int producer_task);
void niles_broadcast_ic_fault(int suspect_pe, int layer_id, unsigned int reporter_pe);
void niles_handle_ic_fault_broadcast(unsigned int source, unsigned int target, unsigned int payload);

void niles_send_heartbeat(unsigned int sender_pe);
void niles_handle_heartbeat(unsigned int source, unsigned int target, unsigned int payload);
void niles_hb_wait_interval(void);

unsigned int niles_stats_bcast_sent(void);
unsigned int niles_stats_bcast_received(void);
unsigned int niles_stats_hb_sent(void);
unsigned int niles_stats_hb_received(void);
unsigned int niles_stats_fault_marks(void);

#endif
