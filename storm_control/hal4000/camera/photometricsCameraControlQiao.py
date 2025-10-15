#!/usr/bin/env python
"""
Camera control specialized for a Photometrics camera.

FIXME? This never calls pvcam.uninitPVCAM() to close the PVCAM
       library. Not sure if this matters.

Hazen 1/18
"""
from PyQt5 import QtCore

import storm_control.sc_hardware.photometrics.pvcam as pvcam
from storm_control.sc_hardware.photometrics import pvcam_constants as C
from storm_control.sc_hardware.photometrics.pvcam import PVCAMException




import storm_control.sc_library.halExceptions as halExceptions
import storm_control.sc_library.parameters as params

import storm_control.hal4000.camera.cameraControl as cameraControl
import storm_control.hal4000.camera.cameraFunctionality as cameraFunctionality


class PhotometricsCameraControl(cameraControl.HWCameraControl):
    """
    This class is used to control a Photometrics camera.
    """
    def __init__(self, config = None, is_master = False, **kwds):
        """
        Create a Photometrics camera control object and initialize the camera.
        """
        kwds["config"] = config
        super().__init__(**kwds)
        self.is_master = is_master
        

        # Load the library and start the camera.
        #
        pvcam.loadPVCAMDLL(config.get("pvcam_sdk"))
        try:
            pvcam.initPVCAM()
        except:
            print("Already Loaded?")

        names = pvcam.getCameraNames()
        if not config.get("camera_name") in names:
            msg = "Camera " + config.get("camera_name") + " is not available. "
            msg += "Available cameras are " + ",".join(str(names)) + "."
            raise halExceptions.HardwareException(msg)
            
        self.camera = pvcam.PVCAMCamera(camera_name = config.get("camera_name"))
        
        # Create the camera functionality.
        #
        have_shutter = self.camera.hasParameter("param_shtr_status")
        have_temperature = self.camera.hasParameter("param_temp_setpoint")
        have_temperature = False
        
        self.camera_functionality = cameraFunctionality.CameraFunctionality(camera_name = self.camera_name,
                                                                            have_shutter = have_shutter,
                                                                            have_temperature = have_temperature,
                                                                            is_master = is_master,
                                                                            parameters = self.parameters)

        # Add Photometrics specific parameters. These all start with 'param_'.
        #
        self.parameters.add(params.ParameterRangeInt(description = "Sensor readout speed",
                                                     name = "param_spdtab_index",
                                                     value = self.camera.getParameterCurrent("param_spdtab_index"),
                                                     min_value = 0,
                                                     max_value = self.camera.getParameterCount("param_spdtab_index") - 1))
                                                     
                                                     
                                                     
                                                     
        ##Qiao 08/24/2025                                      
        # hal4000/camera/photometricsCameraControl.py  (add with the other self.parameters.add(...))
# External edge trigger enable (PVCAM constant PARAM_EDGE_TRIGGER)
       # PVCAM exposure/trigger mode (maps to PARAM_EXPOSURE_MODE)
      #  self.parameters.add(params.ParameterSetString(
           # name="param_exposure_mode",
           # description="PVCAM exposure/trigger mode",
            #value="timed",   # internal/free-run
            #allowed=["timed", "strobed", "bulb", "trigger_first"]))




        # Dictionary of the Photometrics camera properties we'll support.
        #
        self.pvcam_props = {
                                     "exposure_time"      : True,
                                    "param_spdtab_index" : True,
                                    #"param_exposure_mode": True,
                                     "x_bin"              : True,
                                    "x_end"              : True,
                                     "x_start"            : True,
                                     "y_bin"              : True,
                                     "y_end"              : True,
                                     "y_start"            : True,
                                     
                                     # *** Ensure trigger/overlap are pushed to PVCAM ***
                                    # "param_trigger_mode" : True,   # "internal" | "ext_edge" | "ext_bulb"
                                   #  "param_overlap"      : True    # True | False
                                    }
    # skip the generic setParameter for this key                           
                                    
        


        # FIXME: Display needs to realize that the camera bit depth has
        #        changed. For now we're just setting to the maximum.
        #
        #bit_depth = self.camera.getParameterCurrent("param_bit_depth")
        
        self.parameters.setv("max_intensity", 2**16)
                
        # X/Y may be reversed here.
        #
        x_chip = self.camera.getParameterCurrent("param_par_size")
        y_chip = self.camera.getParameterCurrent("param_ser_size")

        # Adjust ranges of our size and binning parameters based on the chip size.
        #
        self.parameters.getp("x_end").setMaximum(x_chip)
        self.parameters.getp("x_start").setMaximum(x_chip)
        self.parameters.getp("y_end").setMaximum(y_chip)
        self.parameters.getp("y_start").setMaximum(y_chip)

        self.parameters.setv("x_end", x_chip)
        self.parameters.setv("y_end", y_chip)
        self.parameters.setv("x_chip", x_chip)
        self.parameters.setv("y_chip", y_chip)

        # FIXME: Need to actually query for possible binning values.
        #
        self.parameters.getp("x_bin").setMaximum(2)
        self.parameters.getp("y_bin").setMaximum(2)

        # Start with 100ms exposure time.
        self.parameters.getp("exposure_time").setOrder(2)
        self.parameters.setv("exposure_time", 0.1)

        self.newParameters(self.parameters, initialization = True)
        
    def _set_exposure_mode(self, human):
        """Set PARAM_EXPOSURE_MODE using pvcam_constants (ints), not strings."""
        mode = (human or "").strip().lower()
        mode_map = {
            "timed":         getattr(C, "TIMED_MODE", None),
            "strobed":       getattr(C, "STROBED_MODE", None),
            "bulb":          getattr(C, "BULB_MODE", None),
            "trigger_first": getattr(C, "TRIGGER_FIRST_MODE", None),
        }
        enum_val = mode_map.get(mode)
        if enum_val is None:
            raise RuntimeError(f"Unsupported param_exposure_mode='{human}' for this PVCAM build")
        self.camera.setParameter("param_exposure_mode", enum_val)
        
        


            
    def newParameters(self, parameters, initialization = False):

        size_x = parameters.get("x_end") - parameters.get("x_start") + 1
        size_y = parameters.get("y_end") - parameters.get("y_start") + 1
        parameters.setv("x_pixels", size_x)
        parameters.setv("y_pixels", size_y)
        parameters.setv("bytes_per_frame", 2 * size_x * size_y)

        super().newParameters(parameters)

        self.camera_working = True

        # Update the parameter values, only the supported ones
        # and only if they are different.
        to_change = []
        for pname in self.pvcam_props:
            if (self.parameters.get(pname) != parameters.get(pname)) or initialization:
                to_change.append(pname)

        if (len(to_change) > 0):
            running = self.running
            if running:
                self.stopCamera()

       ##Qiao 08252025
           # mode = parameters.get("param_exposure_mode")
           # if mode is not None:
            #    self._set_enum_from_string(self.camera, "param_exposure_mode", mode)
             #   applied["param_exposure_mode"] = True
                # skip the generic setParameter for this key




            # Reconfigure the camera.
            #for pname in to_change:
               # if pname.startswith("param_"):
                    #self.camera.setParameter(pname, parameters.get(pname))
            # Reconfigure the camera.
# ---- Apply exposure/trigger mode first (Cam2: 'strobed'), then others ----
           # mode = parameters.get("param_exposure_mode")

            # It’s safest to stop any acquisition before changing mode.
            #try:
            #    self.camera.stopAcquisition()
           # except Exception:
            #    pass

           # if ("param_exposure_mode" in self.pvcam_props) and (mode is not None):
                # Only write non-default modes (PVCAM often denies setting 'timed' here).
             #   if mode.strip().lower() != "timed":
                 #   try:
                  #      self._set_exposure_mode(mode)   # maps string -> enum int and calls setParameter(...)
                  #  except PVCAMException as e:
                  #      print(f"[PVCAM] Could not set exposure_mode='{mode}': {e}")

            # Debug probe: print what PVCAM thinks right now.
           # try:
              #  cur_idx = self.camera.getParameterCurrent("param_exposure_mode")
              #  val, desc = self.camera.getParameterEnum("param_exposure_mode", cur_idx)
              #  print(f"[PVCAM] {self.camera_name} PARAM_EXPOSURE_MODE -> {desc} ({val})")
            #except Exception as e:
              #  print(f"[PVCAM] Enum probe failed: {e}")

            # Apply remaining 'param_*' that are not the mode (e.g., spdtab, etc.)
          #  for pname in to_change:
              #  if (pname != "param_exposure_mode") and pname.startswith("param_"):
                #    self.camera.setParameter(pname, parameters.get(pname))









            # HAL uses seconds, PVCAM uses milliseconds (as integers).
            exposure_time = int(1000.0 * parameters.get("exposure_time"))

            # The camera is zero indexed, but the HAL parameters are
            # one indexed.
            self.camera.captureSetup(parameters.get("x_start")-1,
                                     parameters.get("x_end")-1,
                                     parameters.get("x_bin"),
                                     parameters.get("y_start")-1,
                                     parameters.get("y_end")-1,
                                     parameters.get("y_bin"),
                                     exposure_time)

            # Copy changed parameter values.
            for pname in to_change:
                self.parameters.setv(pname, parameters.get(pname))
            
            # Estimate FPS.
            #
            # FIXME: How to get the actual value for this? For reasons that are
            #        very hard for me to understand Photometrics does not provide
            #        an API function that does this.
            cycle_time = self.parameters.get("exposure_time")
            self.parameters.setv("fps", 1.0/cycle_time)

            # Update bit depth.
            #bit_depth = self.camera.getParameterCurrent("param_bit_depth")
            #self.parameters.setv("max_intensity", 2**bit_depth)

            if running:
                self.startCamera()
                
            self.camera_functionality.parametersChanged.emit()

    # FIXME: For short films we should configure for fixed length acquisition.
    #
    def startFilm(self, film_settings, is_time_base):
        super().startFilm(film_settings, is_time_base)

    def stopFilm(self):
        super().stopFilm()                                                    
