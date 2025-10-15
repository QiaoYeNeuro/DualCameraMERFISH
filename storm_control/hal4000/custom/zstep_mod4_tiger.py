# Z-step after every 4th frame using Tiger AO voltage
# Works with Path B (Cam1 master). Delays the step by after_ms to avoid overlap with exposure.

from PyQt5 import QtCore
import storm_control.hal4000.halLib.halModule as halModule
import storm_control.hal4000.halLib.halMessage as halMessage

def _get_frame_number(msg):
    # Be tolerant to small API differences across hal4000 trees
    data = msg.getData()
    if "frame" in data and hasattr(data["frame"], "frame_number"):
        return data["frame"].frame_number
    if "frame_number" in data:
        return data["frame_number"]
    # Fallback: older messages may just pass an int
    return int(data)

class ZStepMod4Tiger(halModule.HalModule):
    """
    After frames where (n % 4) == 3, wait 'after_ms' then add +dz_um to Tiger AO.
    Parameters (XML):
      dz_um            : step size per z-plane (e.g., 0.5)
      microns_to_volts : Tiger AO scaling (V/μm) or (μm→V factor). If your XML stores μm→V, use that directly.
      start_volts      : starting AO voltage before the stack (e.g., 5.0)
      after_ms         : delay after the 4th frame before stepping (e.g., 3.0)
      ao_func_name     : (optional) exact functionality name to bind, default 'daq.tigerz.ao_task'
    """
    def __init__(self, module_params=None, **kw):
        super().__init__(**kw)
        p = module_params.get("parameters")
        # Required params
        self.dz_um   = float(p.get("dz_um"))
        self.um2V    = float(p.get("microns_to_volts"))
        self.startV  = float(p.get("start_volts"))
        self.after_ms = float(p.get("after_ms"))
        # Optional: allow XML to override the functionality name
        self.ao_name = p.get("ao_func_name") or "daq.tigerz.ao_task"

        self.k = 0     # z-plane index
        self.ao = None # AO functionality handle

    def processMessage(self, message):
        mtype = message.getType()

        if mtype == "configure1":
            # Ask HAL to give us the Tiger AO functionality by name
            self.sendMessage(halMessage.HalMessage(
                m_type = "request functionality",
                data   = {"name": self.ao_name}))

        elif mtype == "functionality":
            # Bind the AO functionality when HAL replies
            if message.getData().get("name") == self.ao_name:
                self.ao = message.getData()["functionality"]
                # Initialize AO to start voltage
                try:
                    self.ao.setVoltage(self.startV)
                except Exception:
                    pass

        elif mtype == "film starting":
            self.k = 0
            if self.ao is not None:
                try:
                    self.ao.setVoltage(self.startV)
                except Exception:
                    pass

        elif mtype == "frame number" and (self.ao is not None):
            n = _get_frame_number(message)
            # After frame indices 3,7,11,... (i.e., n % 4 == 3) do a Z step
            if (n % 4) == 3:
                # Slight delay so we don't move during exposure/readout
                QtCore.QTimer.singleShot(int(self.after_ms), self._step_once)

        elif mtype == "film finished":
            # Optional: park back at startV
            if self.ao is not None:
                try:
                    self.ao.setVoltage(self.startV)
                except Exception:
                    pass

    def _step_once(self):
        self.k += 1
        try:
            self.ao.setVoltage(self.startV + self.k * self.dz_um * self.um2V)
        except Exception:
            pass
