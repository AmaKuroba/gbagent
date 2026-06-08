#!/bin/bash
cd ~/projects/gbagent

hermes kanban --board gbagent create \
  --skill test-driven-development \
  "CPU: Core execution engine" \
  --priority 1 \
  --body "Implement the LR35902 CPU core with fetch-decode-execute.

Design:
- Core struct wraps MemoryBus (Read/Write at uint16 addr)
- Step() fetches byte at PC, decodes via existing Decode(opcode), advances PC
- Already: CPUState struct, Decode/DecodeCB functions
- Bus test stub: type Bus struct { data [0x10000]byte } with Read/Write methods

TDD:
1. Write cpu_core_test.go with TestExecuteNOP, TestExecuteLD_BC_d16, TestExecuteLD_A_d8
2. Implement cpu_core.go with Core struct + Step()
3. Verify: go test ./internal/gb/ -run TestExecute -v -count=1

Files: internal/gb/cpu_core.go, internal/gb/cpu_core_test.go (write test FIRST)"

echo "Task 1 created: $?"
