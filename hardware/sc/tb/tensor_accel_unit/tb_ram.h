#ifndef TB_RAM_H_
#define TB_RAM_H_

#include "../../standards.h"
#include <cstring>

/*
 * Minimal RAM shadow for tensor_accel unit TB (no PE ports / load_ram).
 * Same ram_data[] layout the accelerator indexes via byte_addr >> 2.
 */
#ifndef RAM_SIZE
#define RAM_SIZE MEMORY_SIZE_BYTES
#endif

class ram {
public:
	unsigned long ram_data[RAM_SIZE];

	ram() {
		std::memset(ram_data, 0, sizeof(ram_data));
	}
};

#endif
