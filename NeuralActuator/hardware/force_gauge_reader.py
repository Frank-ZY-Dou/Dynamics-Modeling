#!/usr/bin/env python3
"""Serial reader for the BAOSHISHAN ZP-500N force gauge.

Requires pyserial (hardware-side extra, not in the top-level requirements.txt).
"""
import argparse
import os
import sys
import time
import re
# from typing import Generator, Optional, Tuple

import serial


class ForceGaugeReader:
    """
    Minimal BAOSHISHAN (and similar) force gauge reader.

    - Defaults to 2400 baud, 7E1 (7 data bits, EVEN parity, 1 stop bit).
    - Parses complete text frames and extracts a decimal with fixed places.
    - Optional 'on_change' de-duplication and plausibility range.

    Usage as a script:
      python gauge_reader.py --unit N --on-change

    Usage as a library:
      with ForceGaugeReader() as g:
          for v in g.stream_values():
              print(v)
    """
    def __init__(self,timeout) -> None:
        self.timeout = 0.1
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
            while True and (time.time() - beg) < 10:
                
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














    # def main(self, once=False) -> int:
    #     port = self.args.port or self.detect_default_port()
    #     if not port:
    #         print("Could not find a serial port. Use --port.", file=sys.stderr)
    #         return 2

    #     try:
    #         ser = serial.Serial(
    #             port=port,
    #             baudrate=self.args.baud,
    #             bytesize=serial.EIGHTBITS,
    #             parity=serial.PARITY_NONE,
    #             stopbits=serial.STOPBITS_ONE,
    #             timeout=0.0,                  # non-blocking
    #             write_timeout=0.1,
    #         )
    #     except Exception as exc:
    #         print(f"Failed to open {port}: {exc}", file=sys.stderr)
    #         return 1

    #     pattern = re.compile(rf"[+-]?\d+\.\d{{{max(0, self.args.decimals)}}}")
    #     buf = ""
    #     last_printed = None
    #     try:
    #         ser.reset_input_buffer()
    #         deadline = time.monotonic() + max(getattr(self.args, "deadline", 1.0), 0.05)  # used only if once=True
    #         while True:
    #             n = ser.in_waiting
    #             if n:
    #                 chunk = ser.read(n)
    #                 if chunk:
    #                     buf += chunk.decode(self.args.encoding, errors="ignore")

    #                     m = pattern.search(buf)   # just need the first match
    #                     if not m:
    #                         buf = buf[-16:]       # keep buffer small
    #                         continue

    #                     text = m.group(0)
    #                     end_pos = m.end()
    #                     buf = buf[end_pos:][-16:]

    #                     try:
    #                         num = float(text)
    #                         out = f"{num:.{self.args.decimals}f}"
    #                     except Exception:
    #                         out = text

    #                     if self.args.on_change and last_printed is not None and out == last_printed:
    #                         # if only once, keep waiting for a changed value until deadline
    #                         if once and time.monotonic() > deadline:
    #                             return out        # or return None to indicate no change before deadline
    #                         continue

    #                     last_printed = out
    #                     if once:
    #                         return out
    #                     else:
    #                         print(f"{out} {self.args.unit}" if self.args.unit else out)
    #             else:
    #                 # no bytes right now
    #                 if once and time.monotonic() > deadline:
    #                     return None
    #                 time.sleep(0.003)             # short nap to reduce CPU
    #     finally:
    #         ser.close()


#     # ---- Construction / config ------------------------------------------------

#     @staticmethod
#     def detect_default_port() -> Optional[str]:
#         candidates = ["/dev/ttyUSB1"]
#         for path in candidates:
#             if os.path.exists(path):
#                 return path
#         return None

#     def __init__(
#         self,
#         port: Optional[str] = None,
#         baud: int = 2400,
#         timeout: float = 1.0,
#         encoding: str = "utf-8",
#         decimals: int = 3,
#         on_change: bool = False,
#         unit: Optional[str] = None,
#         plausible: Tuple[Optional[float], Optional[float]] = (None, None),
#         # framing defaults to 7E1 which many gauges use; can be overridden
#         bytesize: int = serial.SEVENBITS,
#         parity: str = serial.PARITY_EVEN,
#         stopbits: int = serial.STOPBITS_ONE,
#         read_terminator: bytes = b"\n",   # switch to b"\r" if your device uses CR only
#     ):
#         self.port = port or self.detect_default_port()
#         # print(self.port, "port")
#         if not self.port:
#             raise RuntimeError("Could not find a serial port. Pass --port explicitly.")

#         self.baud = baud
#         self.timeout = timeout
#         self.encoding = encoding
#         self.decimals = max(0, int(decimals))
#         self.on_change = on_change
#         self.unit = unit
#         self.plausible = plausible
#         self.bytesize = bytesize
#         self.parity = parity
#         self.stopbits = stopbits
#         self.read_terminator = read_terminator

#         # Boundary-aware number with exact decimal places
#         self._pattern = re.compile(
#             rf"(?<![\d.-])[+-]?\d+\.\d{{{self.decimals}}}(?![\d.])"
#         )

#         self._ser: Optional[serial.Serial] = None
#         self._last_out: Optional[str] = None

#     # ---- Context manager ------------------------------------------------------

#     def __enter__(self):
#         self.open()
#         return self

#     def __exit__(self, exc_type, exc, tb):
#         self.close()

#     # ---- Serial I/O -----------------------------------------------------------

#     def open(self):
#         if self._ser and self._ser.is_open:
#             return
#         try:
#             self._ser = serial.Serial(
#                 port=self.port,
#                 baudrate=self.baud,
#                 bytesize=self.bytesize,
#                 parity=self.parity,
#                 stopbits=self.stopbits,
#                 timeout=self.timeout,
#             )
#             time.sleep(0.2)
#             self._ser.reset_input_buffer()
#         except Exception as exc:
#             raise RuntimeError(f"Failed to open {self.port}: {exc}") from exc

#     def close(self):
#         if self._ser:
#             try:
#                 self._ser.close()
#             except Exception:
#                 pass
#             finally:
#                 self._ser = None

#     def _readline(self) -> Optional[bytes]:
#         """Read until terminator or timeout. Returns bytes or None on timeout."""
#         if not self._ser:
#             return None
#         # If your device sends only CR, set read_terminator=b"\r"
#         line = self._ser.read_until(self.read_terminator)
#         return line if line else None

#     # ---- Parsing / streaming --------------------------------------------------

#     def parse_value_from_text(self, text: str) -> Optional[float]:
#         m = self._pattern.search(text)
#         if not m:
#             return None
#         try:
#             return float(m.group(0))
#         except ValueError:
#             return None

#     def _passes_plausibility(self, val: float) -> bool:
#         lo, hi = self.plausible
#         if lo is not None and val < lo:
#             return False
#         if hi is not None and val > hi:
#             return False
#         return True

#     def stream_values(self) -> Generator[str, None, None]:
#         """
#         Yields normalized strings like '0.123' with exactly `decimals` places.
#         Respects on_change and plausibility range if configured.
#         """
#         if not self._ser or not self._ser.is_open:
#             print("1")
#             self.open()

#         while True:
#             print("2")
#             raw = self._readline()
#             print("3")
#             if not raw:
#                 continue
#             try:
#                 print("4")
#                 line = raw.decode(self.encoding, errors="ignore").strip()
#             except Exception:
#                 continue
#             print("5")
#             val = self.parse_value_from_text(line)

#             if val is None:
#                 continue
#             print("6")
#             if not self._passes_plausibility(val):
#                 continue
#             print("7")
#             out = f"{val:.{self.decimals}f}"
#             if self.on_change and self._last_out is not None and out == self._last_out:
#                 continue
#             self._last_out = out
#             yield out

#     # ---- One-shot helpers -----------------------------------------------------

#     # def read_one(self) -> Optional[str]:
#     #     """
#     #     Return exactly one parsed/normalized value, or None if not available before timeout.
#     #     """
#     #     deadline = time.time() + max(self.timeout, 0.01)
#     #     # print("this is the problem")
#     #     # for out in self.stream_values():
#     #     #     print("out")
#     #     #     return out
#     #     # print("not out")
#     #         # unreachable, but kept for clarity
#     #     # If stream_values never yielded (e.g., timeout looped), respect deadline
#     #     while time.time() < deadline:
#     #         # print("force")
#     #         raw = self._readline()
#     #         if not raw:
#     #             continue
#     #         line = raw.decode(self.encoding, errors="ignore").strip()
#     #         val = self.parse_value_from_text(line)
#     #         if val is None or not self._passes_plausibility(val):
#     #             continue
#     #         return f"{val:.{self.decimals}f}"
#     #     return None
#     def read_one(self, deadline_s: float = None):
#         import time
#         if deadline_s is None:
#             deadline_s = max(self.timeout, 0.1)  # give it a little time

#         deadline = time.time() + deadline_s
#         while time.time() < deadline:
#             raw = self._readline()
#             if not raw:
#                 continue
#             try:
#                 line = raw.decode(self.encoding, errors="ignore").strip()
#             except Exception as e:
#                 print("decode error:", e)
#                 continue
#             print(f"[DEBUG] line: {repr(line)}")   # show EXACT text
#             val = self.parse_value_from_text(line)
#             print(f"[DEBUG] parsed: {val!r}")
#             return val  # return raw parsed value without plausibility/on_change/rounding
#         print("[DEBUG] no data before deadline")
#         return None


#     # ---- CLI ------------------------------------------------------------------

#     @classmethod
#     def main_from_args(cls) -> int:
#         parser = argparse.ArgumentParser(description="BAOSHISHAN gauge reader (2400 bps ASCII)")
#         parser.add_argument("--port", default=None, help="Serial port, e.g. /dev/ttyUSB0")
#         parser.add_argument("--baud", type=int, default=2400, help="Baud rate (default: 2400)")
#         parser.add_argument("--timeout", type=float, default=1.0, help="Read timeout seconds")
#         parser.add_argument("--encoding", default="utf-8", help="Decode encoding (default: utf-8)")
#         parser.add_argument("--decimals", type=int, default=3, help="Expected decimal places (default: 3)")
#         parser.add_argument("--on-change", action="store_true", help="Only print when value changes")
#         parser.add_argument("--unit", default=None, help="Append unit when printing, e.g. N")
#         parser.add_argument("--min", dest="minval", type=float, default=None, help="Minimum plausible value")
#         parser.add_argument("--max", dest="maxval", type=float, default=None, help="Maximum plausible value")
#         parser.add_argument("--once", action="store_true", help="Read and print exactly one value, then exit")
#         parser.add_argument("--cr", action="store_true", help="Use CR (\\r) as line terminator instead of LF")
#         parser.add_argument("--eightn1", action="store_true",
#                             help="Use 8N1 framing instead of default 7E1")
#         self.args = parser.parse_args()

#         try:
#             reader = cls(
#                 port=self.args.port,
#                 baud=self.args.baud,
#                 timeout=self.args.timeout,
#                 encoding=self.args.encoding,
#                 decimals=self.args.decimals,
#                 on_change=self.args.on_change,
#                 unit=self.args.unit,
#                 plausible=(self.args.minval, self.args.maxval),
#                 bytesize=serial.EIGHTBITS if self.args.eightn1 else serial.SEVENBITS,
#                 parity=serial.PARITY_NONE if self.args.eightn1 else serial.PARITY_EVEN,
#                 stopbits=serial.STOPBITS_ONE,
#                 read_terminator=(b"\r" if self.args.cr else b"\n"),
#             )
#         except RuntimeError as e:
#             print(str(e), file=sys.stderr)
#             return 2

#         framing = "8N1" if self.args.eightn1 else "7E1"
#         print(f"Opened {reader.port} @ {reader.baud} baud ({framing}). Ctrl+C to stop.")

#         try:
#             with reader:
#                 if self.args.once:
#                     val = reader.read_one()
#                     if val is None:
#                         return 1
#                     print(f"{val} {self.args.unit}" if self.args.unit else val)
#                     return 0
#                 for val in reader.stream_values():
#                     print(f"{val} {self.args.unit}" if self.args.unit else val)
#         except KeyboardInterrupt:
#             pass
#         finally:
#             reader.close()
#         return 0


# if __name__ == "__main__":
#     sys.exit(ForceGaugeReader.main_from_args())
