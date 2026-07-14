#!/usr/bin/env python3
import argparse
import os
import sys
import time
import re

import serial


class ForceGaugeReader:

    def __init__(self,timeout) -> None:
        self.timeout = timeout
        parser = argparse.ArgumentParser(description="Minimal BAOSHISHAN gauge reader (2400 bps ASCII)")
        parser.add_argument("--port", default=None, help="Serial port, e.g. /dev/ttyUSB0")
        parser.add_argument("--baud", type=int, default=2400, help="Baud rate (default: 2400)")
        parser.add_argument("--timeout", type=float, default=0.01, help="Read timeout seconds")
        parser.add_argument("--encoding", default="utf-8", help="Decode encoding (default: utf-8)")
        parser.add_argument("--decimals", type=int, default=3, help="Expected decimal places (default: 3)")
        parser.add_argument("--on-change", action="store_true", help="Only print when value changes")
        parser.add_argument("--unit", default=None, help="Append unit when printing, e.g. N")
        self.args = parser.parse_args()
        self.port = self.args.port or self.detect_default_port()
        if not self.port:
            print("Could not find a serial port. Use --port.", file=sys.stderr)
            return 2


    def detect_default_port(self):
        candidates = ["/dev/ttyUSB1"]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None


    def main(self, once = False) -> int:
        
        

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.args.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.args.timeout,
            )
        except Exception as exc:
            print(f"Failed to open {self.port}: {exc}", file=sys.stderr)
            return 1
        
        # print(f"Opened {port} @ {self.args.baud} baud. Ctrl+C to stop.")
        
        pattern = re.compile(rf"[+-]?\d+\.\d{{{max(0, self.args.decimals)}}}")
        buf = ""
        last_printed: str | None = None
        try:
            
            # time.sleep(0.2)
            
            self.ser.reset_input_buffer()
            beg = time.time()
            # count = 0
            while True and (time.time() - beg) < 0.1:
                
                # count += 1
                chunk = self.ser.read(128)
                # if count < 10:
                #     print(time.time() - beg, "time taken")
                if not chunk:
                    continue
                buf += chunk.decode(self.args.encoding, errors="ignore")
                matches = list(pattern.finditer(buf))
                if not matches:
                    buf = buf[-8:]
                    continue
                end_pos = 0
                for m in matches:
                    text = m.group(0)
                    end_pos = max(end_pos, m.end())
                    # Normalize: cast to float to drop extra leading zeros
                    try:
                        num = float(text)
                        out = f"{num:.{self.args.decimals}f}"
                    except Exception:
                        out = text
                    if self.args.on_change and last_printed is not None and out == last_printed:
                        print("here")
                        continue
                    last_printed = out
                    if once is True:
                        # print(count, "count")
                        # print(time.time() - beg, "time taken")
                        return out
                    else:
                        print(f"{out} {self.args.unit}" if self.args.unit else out)
                buf = buf[end_pos:][-8:]
        finally:
            self.ser.close()
        # return 0
