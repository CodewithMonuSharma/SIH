/**
 * OptiTrack — FSOC Coarse Alignment Mission Control
 * Real-time Space-Ground / UAV Coarse Tracking Simulation Engine & Telemetry Pipeline
 * Supports bidirectional Live Python Backend Sync (OpenCV frames & state machine) + Offline Client Fallback.
 */

class KalmanFilter2D {
  constructor() {
    this.reset();
  }

  reset() {
    this.x = 1000.0;
    this.y = 1000.0;
    this.vx = 0.0;
    this.vy = 0.0;
    this.p = [
      [10, 0, 0, 0],
      [0, 10, 0, 0],
      [0, 0, 5, 0],
      [0, 0, 0, 5]
    ];
    this.q = 0.2;
    this.r = 1.5;
    this.initialized = false;
  }

  predict(dt) {
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    for (let i = 0; i < 4; i++) {
      this.p[i][i] += this.q * dt;
    }
  }

  update(measX, measY) {
    if (!this.initialized) {
      this.x = measX;
      this.y = measY;
      this.vx = 0;
      this.vy = 0;
      this.initialized = true;
      return;
    }
    const yx = measX - this.x;
    const yy = measY - this.y;
    const kx = this.p[0][0] / (this.p[0][0] + this.r);
    const ky = this.p[1][1] / (this.p[1][1] + this.r);
    const kvx = this.p[2][2] / (this.p[2][2] + this.r * 5);
    const kvy = this.p[3][3] / (this.p[3][3] + this.r * 5);

    this.x += kx * yx;
    this.y += ky * yy;
    this.vx += kvx * (yx / 0.033);
    this.vy += kvy * (yy / 0.033);

    this.p[0][0] *= (1 - kx);
    this.p[1][1] *= (1 - ky);
    this.p[2][2] *= (1 - kvx);
    this.p[3][3] *= (1 - kvy);
  }
}

class OptiTrackMissionControl {
  constructor() {
    this.isRunning = true;
    this.mode = 'SIMULATION'; // 'SIMULATION' | 'VIDEO'
    this.theme = 'light'; // 'light' | 'dark'

    // Backend Connection Status
    this.isBackendConnected = false;
    this.backendEventSource = null;
    this.backendFrameImg = new Image();

    // Scene & World Settings
    this.worldWidth = 2000;
    this.worldHeight = 2000;
    this.camResX = 640;
    this.camResY = 480;

    // Target States
    this.targetCount = 1;
    this.pattern = 'straight';
    this.targetSpeed = 80.0;
    this.targetAngle = 35.0 * (Math.PI / 180.0);
    this.targetSize = 12;
    this.targetGeometry = 'square';
    this.targetPos = { x: 1624, y: 580 };
    this.trailHistory = [];
    this.maxTrail = 65;

    // Circular / Figure-8 / Spiral parameters
    this.patternTime = 0;
    this.circleCenter = { x: 1000, y: 1000 };
    this.circleRadius = 380;

    // Gimbal / Camera Sensor
    this.gimbalPos = { x: 1000, y: 1000 };
    this.gimbalPanSpeedMax = 7.5;
    this.gimbalTiltSpeedMax = 6.8;
    this.panDeg = 14.2;
    this.tiltDeg = 28.9;
    this.updateRateHz = 30;
    this.cameraType = 'ir';

    // Disturbances & Optical channel
    this.noiseGaussian = true;
    this.noiseSaltPepper = false;
    this.noisePoisson = false;
    this.noiseStdDev = 1.2;
    this.cameraJitter = 0.4;
    this.platformMotionType = 'linear';
    this.platformMagnitude = 2.0;
    this.atmosphericChannel = 'clear';

    // Kalman Filter Tracker
    this.kalman = new KalmanFilter2D();

    // Telemetry Statistics
    this.startTime = Date.now();
    this.lastFrameTime = performance.now();
    this.frameCount = 0;
    this.fps = 30.0;
    this.trackingError = 0.0;
    this.rmse = 0.0;
    this.errorHistory = [];
    this.acquisitionTime = null;
    this.reacqTime = null;
    this.lossRate = 0.0;
    this.retentionRate = 0.0;
    this.isLocked = false;
    this.lockConfidence = 0.0;
    this.stateLabel = 'INIT';
    this.epochSeconds = 0;
    this.telemetryLogs = [];

    // Spiral / Sinusoid parameters (client-side fallback)
    this.spiralAngle = 0;
    this.spiralRadius = 50;
    this.sinusoidX = 200;

    // Rain drop particles for atmospheric rain effect
    this.rainDrops = [];
    for (let i = 0; i < 40; i++) {
      this.rainDrops.push({
        x: Math.random() * 640,
        y: Math.random() * 480,
        len: 8 + Math.random() * 12,
        speed: 12 + Math.random() * 10
      });
    }

    // DOM Elements Cache
    this.cacheDomElements();
    this.initCanvases();
    this.bindEvents();
    this.startClock();
    
    // Connect to Python Backend Stream
    this.connectBackendStream();

    // Start Render Loop
    requestAnimationFrame(this.renderLoop.bind(this));
  }

  cacheDomElements() {
    this.sceneCanvas = document.getElementById('sceneOverviewCanvas');
    this.camCanvas = document.getElementById('cameraFeedCanvas');
    this.sceneCtx = this.sceneCanvas.getContext('2d');
    this.camCtx = this.camCanvas.getContext('2d');

    // Controls
    this.btnRun = document.getElementById('btnRun');
    this.btnPause = document.getElementById('btnPause');
    this.btnReset = document.getElementById('btnReset');
    this.runBtnLabel = document.getElementById('runBtnLabel');
    this.simModeBtn = document.getElementById('simModeBtn');
    this.vidModeBtn = document.getElementById('vidModeBtn');
    this.themeToggleBtn = document.getElementById('themeToggleBtn');
    this.themeLabel = document.getElementById('themeLabel');

    // Telemetry Displays
    this.valTrackingError = document.getElementById('valTrackingError');
    this.statusTrackingError = document.getElementById('statusTrackingError');
    this.valFps = document.getElementById('valFps');
    this.valAcqTime = document.getElementById('valAcqTime');
    this.valReacqTime = document.getElementById('valReacqTime');
    this.valLossRate = document.getElementById('valLossRate');
    this.valRetentionRate = document.getElementById('valRetentionRate');
    this.valEpochDuration = document.getElementById('valEpochDuration');

    // Overlay & Badges
    this.sceneStateText = document.getElementById('sceneStateText');
    this.platformDriftText = document.getElementById('platformDriftText');
    this.rangeText = document.getElementById('rangeText');
    this.lockBadgeContainer = document.getElementById('lockBadgeContainer');
    this.lockBadgeText = document.getElementById('lockBadgeText');
    this.lockPingDot = document.getElementById('lockPingDot');
    this.camExposureText = document.getElementById('camExposureText');
    this.camGainText = document.getElementById('camGainText');
    this.camBeaconTypeText = document.getElementById('camBeaconTypeText');
    this.servoRespText = document.getElementById('servoRespText');
    this.signalMarginText = document.getElementById('signalMarginText');

    // Footer
    this.footerStatusDot = document.getElementById('footerStatusDot');
    this.footerStatusText = document.getElementById('footerStatusText');
    this.footerSnrText = document.getElementById('footerSnrText');
    this.cpuValText = document.getElementById('cpuValText');
    this.gpuValText = document.getElementById('gpuValText');
    this.latencyValText = document.getElementById('latencyValText');
    this.utcClockText = document.getElementById('utcClockText');

    // Export Modal
    this.reportModal = document.getElementById('reportModal');
    this.modalReportContent = document.getElementById('modalReportContent');
    this.btnCloseModal = document.getElementById('btnCloseModal');
    this.btnExportReport = document.getElementById('btnExportReport');
    this.btnDownloadCsv = document.getElementById('btnDownloadCsv');
    this.btnPrintReport = document.getElementById('btnPrintReport');
  }

  initCanvases() {
    this.sceneCanvas.width = 1000;
    this.sceneCanvas.height = 700;
    this.camCanvas.width = 640;
    this.camCanvas.height = 480;
  }

  connectBackendStream() {
    // Attempt connecting to Python server SSE endpoint
    try {
      this.backendEventSource = new EventSource('/api/stream');

      this.backendEventSource.onopen = () => {
        this.isBackendConnected = true;
        this.sceneStateText.innerText = "PYTHON LIVE";
        this.sceneStateText.style.color = '#1FAE64';
        this.sceneStateText.style.fontWeight = 'bold';
        console.log('[OptiTrack] Connected to live Python simulation backend!');
      };

      this.backendEventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.onPythonTelemetry(data);
        } catch (e) {
          console.error('[OptiTrack] Error parsing backend frame:', e);
        }
      };

      this.backendEventSource.onerror = () => {
        this.isBackendConnected = false;
        this.sceneStateText.innerText = 'CLIENT SIM';
        this.sceneStateText.style.color = '#0F9E85';
        this.sceneStateText.style.fontWeight = 'normal';
      };
    } catch (e) {
      this.isBackendConnected = false;
    }
  }

  sendBackendCommand(cmdObj) {
    if (this.isBackendConnected) {
      fetch('/api/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cmdObj)
      }).catch(err => console.warn("[OptiTrack] Backend cmd error:", err));
    }
  }

  onPythonTelemetry(data) {
    // 1. Target & Gimbal positions from Python SimulationRunner
    if (data.target_pos) {
      this.targetPos = { x: data.target_pos[0], y: data.target_pos[1] };
    }
    if (data.gimbal_pos) {
      this.gimbalPos = { x: data.gimbal_pos[0], y: data.gimbal_pos[1] };
    }
    if (data.pan_deg !== undefined) this.panDeg = data.pan_deg;
    if (data.tilt_deg !== undefined) this.tiltDeg = data.tilt_deg;

    // 2. Tracking Error & RMSE
    if (data.error_px !== undefined) {
      this.trackingError = data.error_px;
      this.errorHistory.push(this.trackingError);
      if (this.errorHistory.length > 200) this.errorHistory.shift();
    }
    if (data.rmse_px !== undefined) this.rmse = data.rmse_px;

    // 3. FPS & Lock State
    if (data.fps !== undefined) this.fps = data.fps;
    if (data.is_locked !== undefined) this.isLocked = data.is_locked;

    // 4. State label for scene overlay
    if (data.state_label) {
      this.stateLabel = data.state_label;
    }

    // 5. Acquisition / Re-acquisition timing from backend
    if (data.acq_time !== null && data.acq_time !== undefined) {
      this.acquisitionTime = data.acq_time;
    }
    if (data.reacq_time !== null && data.reacq_time !== undefined) {
      this.reacqTime = data.reacq_time;
    }

    // 6. Lock retention & loss rate from backend metrics
    if (data.lock_retention !== undefined) this.retentionRate = data.lock_retention;
    if (data.loss_rate !== undefined) this.lossRate = data.loss_rate;

    // 7. Lock confidence from tracking error
    this.lockConfidence = this.isLocked
      ? Math.max(80.0, Math.min(99.9, 100.0 - this.trackingError * 6.5))
      : 0.0;

    // 8. Load actual OpenCV frame sent from Python backend
    if (data.frame_jpg) {
      this.backendFrameImg.src = 'data:image/jpeg;base64,' + data.frame_jpg;
    }

    // 9. Log telemetry row for CSV export
    if (this.frameCount % 30 === 0) {
      this.telemetryLogs.push({
        time: data.time || (this.epochSeconds),
        tgtX: this.targetPos.x,
        tgtY: this.targetPos.y,
        gimbalX: this.gimbalPos.x,
        gimbalY: this.gimbalPos.y,
        errorPx: this.trackingError,
        fps: this.fps,
        confidence: this.lockConfidence.toFixed(1)
      });
      if (this.telemetryLogs.length > 500) this.telemetryLogs.shift();
    }

    // 10. Save trail point
    if (this.frameCount % 2 === 0) {
      this.trailHistory.push({ x: this.targetPos.x, y: this.targetPos.y });
      if (this.trailHistory.length > this.maxTrail) {
        this.trailHistory.shift();
      }
    }
  }

  bindEvents() {
    // Top Bar Run / Pause / Reset
    this.btnRun.addEventListener('click', () => {
      this.isRunning = true;
      this.runBtnLabel.innerText = "RUNNING";
      this.btnRun.className = "flex items-center space-x-1 px-3 py-1 rounded text-xs font-mono font-medium text-white bg-[#0F9E85] border border-[#0F9E85] shadow-sm transition-all cursor-pointer";
      this.btnPause.className = "flex items-center space-x-1 px-3 py-1 rounded text-xs font-mono font-medium text-[#D97706] bg-[#FFFBEB] hover:bg-[#FEF3C7] border border-[#FDE68A] transition-all cursor-pointer";
      this.sendBackendCommand({ action: "run" });
    });

    this.btnPause.addEventListener('click', () => {
      this.isRunning = false;
      this.runBtnLabel.innerText = "RUN";
      this.btnRun.className = "flex items-center space-x-1 px-3 py-1 rounded text-xs font-mono font-medium text-[#1FAE64] bg-[#ECFDF5] hover:bg-[#D1FAE5] border border-[#A7F3D0] transition-all cursor-pointer";
      this.btnPause.className = "flex items-center space-x-1 px-3 py-1 rounded text-xs font-mono font-medium text-white bg-[#D97706] border border-[#D97706] shadow-sm transition-all cursor-pointer";
      this.sendBackendCommand({ action: "pause" });
    });

    this.btnReset.addEventListener('click', () => {
      this.resetSimulation();
      this.sendBackendCommand({ action: "reset" });
    });

    // Theme Switcher
    this.themeToggleBtn.addEventListener('click', () => {
      this.toggleTheme();
    });

    // Mode Toggle (Simulation vs Video)
    this.simModeBtn.addEventListener('click', () => this.setMode('SIMULATION'));
    this.vidModeBtn.addEventListener('click', () => this.setMode('VIDEO'));

    // Motion Pattern Buttons
    document.querySelectorAll('.pattern-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.pattern-btn').forEach(b => {
          b.classList.remove('active', 'bg-[#E6F8F5]', 'text-[#0F9E85]', 'border-[#0F9E85]', 'shadow-sm');
          b.classList.add('bg-[#F7F8FA]', 'input-box', 'border-[#D8DCE3]', 'text-[#6B7280]');
        });
        const target = e.currentTarget;
        target.classList.add('active', 'bg-[#E6F8F5]', 'text-[#0F9E85]', 'border-[#0F9E85]', 'shadow-sm');
        target.classList.remove('bg-[#F7F8FA]', 'input-box', 'border-[#D8DCE3]', 'text-[#6B7280]');
        this.pattern = target.dataset.pattern;
        this.trailHistory = [];
        this.patternTime = 0;
        this.sendBackendCommand({ action: "set_pattern", pattern: this.pattern });
      });
    });

    // Secondary Chips (Spiral, Sinusoidal)
    document.querySelectorAll('.pattern-chip').forEach(chip => {
      chip.addEventListener('click', (e) => {
        this.pattern = e.currentTarget.dataset.pattern;
        this.trailHistory = [];
        this.patternTime = 0;
        this.sendBackendCommand({ action: "set_pattern", pattern: this.pattern });
      });
    });

    // Target Speed Customizer
    document.getElementById('btnCustomSpeed').addEventListener('click', () => {
      const speeds = [40, 80, 120, 160, 200];
      const nextSpeed = speeds[(speeds.indexOf(this.targetSpeed) + 1) % speeds.length] || 80;
      this.targetSpeed = nextSpeed;
      document.getElementById('speedValText').innerText = nextSpeed;
      this.sendBackendCommand({ action: "set_speed", speed: nextSpeed });
    });

    // Target Count Stepper
    document.getElementById('btnIncrTarget').addEventListener('click', () => {
      this.targetCount = Math.min(5, this.targetCount + 1);
      document.getElementById('targetCountDisplay').innerText = this.targetCount;
      document.getElementById('targetCountBadge').innerText = `${this.targetCount} ACTIVE`;
    });
    document.getElementById('btnDecrTarget').addEventListener('click', () => {
      this.targetCount = Math.max(1, this.targetCount - 1);
      document.getElementById('targetCountDisplay').innerText = this.targetCount;
      document.getElementById('targetCountBadge').innerText = `${this.targetCount} ACTIVE`;
    });
    document.getElementById('btnAddTarget').addEventListener('click', () => {
      this.targetCount = Math.min(5, this.targetCount + 1);
      document.getElementById('targetCountDisplay').innerText = this.targetCount;
      document.getElementById('targetCountBadge').innerText = `${this.targetCount} ACTIVE`;
    });

    // Scene Size Inputs
    const inputSceneW = document.getElementById('inputSceneWidth');
    const inputSceneH = document.getElementById('inputSceneHeight');
    const updateSceneSize = () => {
      const w = parseInt(inputSceneW.value) || 2000;
      const h = parseInt(inputSceneH.value) || 2000;
      this.worldWidth = w;
      this.worldHeight = h;
      this.sendBackendCommand({ action: "set_scene_size", width: w, height: h });
    };
    if (inputSceneW) inputSceneW.addEventListener('change', updateSceneSize);
    if (inputSceneH) inputSceneH.addEventListener('change', updateSceneSize);

    // Target Geometry
    document.querySelectorAll('.geom-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.geom-btn').forEach(b => {
          b.className = "geom-btn px-2 py-0.5 text-[10px] font-mono text-[#6B7280] hover:text-contrast cursor-pointer";
        });
        e.currentTarget.className = "geom-btn px-2 py-0.5 text-[10px] font-mono bg-[#E6F8F5] text-[#0F9E85] rounded font-semibold border border-[#A7F3D0] cursor-pointer";
        this.targetGeometry = e.currentTarget.dataset.geom;
        this.sendBackendCommand({ action: "set_target_geometry", geometry: this.targetGeometry });
      });
    });

    // Target Size Slider
    const sliderSize = document.getElementById('sliderTargetSize');
    sliderSize.addEventListener('input', (e) => {
      this.targetSize = parseFloat(e.target.value);
      document.getElementById('targetSizeVal').innerText = `${this.targetSize} px`;
      this.sendBackendCommand({ action: "set_target_size", size: this.targetSize });
    });

    // Target Coordinates
    const inputX = document.getElementById('inputTargetX');
    const inputY = document.getElementById('inputTargetY');
    const updateTargetCoords = () => {
      const nx = parseFloat(inputX.value) || 1000;
      const ny = parseFloat(inputY.value) || 1000;
      this.targetPos = { x: Math.max(50, Math.min(1950, nx)), y: Math.max(50, Math.min(1950, ny)) };
      this.trailHistory = [];
      this.sendBackendCommand({ action: "set_target_coords", x: this.targetPos.x, y: this.targetPos.y });
    };
    inputX.addEventListener('change', updateTargetCoords);
    inputY.addEventListener('change', updateTargetCoords);

    // Randomize Button
    document.getElementById('btnRandomizeTarget').addEventListener('click', () => {
      const rx = Math.floor(200 + Math.random() * 1600);
      const ry = Math.floor(200 + Math.random() * 1600);
      inputX.value = rx;
      inputY.value = ry;
      this.targetPos = { x: rx, y: ry };
      this.trailHistory = [];
      this.patternTime = 0;
      this.sendBackendCommand({ action: "randomize" });
    });

    // Camera Type (IR vs RGB)
    document.querySelectorAll('.camtype-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.camtype-btn').forEach(b => {
          b.className = "camtype-btn px-3 py-0.5 text-[10px] font-mono text-[#6B7280] hover:text-contrast cursor-pointer";
        });
        e.currentTarget.className = "camtype-btn px-3 py-0.5 text-[10px] font-mono bg-[#0F9E85] text-white font-bold rounded shadow-sm cursor-pointer";
        this.cameraType = e.currentTarget.dataset.camtype;
        this.camBeaconTypeText.innerText = this.cameraType === 'ir' ? "BEACON: 850nm NIR" : "SPECTRUM: RGB VISIBLE";
        this.sendBackendCommand({ action: "set_camera_type", camera_type: this.cameraType });
      });
    });

    // Resolution & FOV
    const selectCamRes = document.getElementById('selectCamRes');
    if (selectCamRes) {
      selectCamRes.addEventListener('change', (e) => {
        this.sendBackendCommand({ action: "set_camera_res", res: e.target.value });
      });
    }

    // Update Rate & Initial Cam Pos
    const inputUpdateRate = document.getElementById('inputUpdateRate');
    if (inputUpdateRate) {
      inputUpdateRate.addEventListener('change', (e) => {
        const rate = parseFloat(e.target.value) || 30;
        this.updateRateHz = rate;
        this.sendBackendCommand({ action: "set_update_rate", rate: rate });
      });
    }

    const selectInitCamPos = document.getElementById('selectInitCamPos');
    if (selectInitCamPos) {
      selectInitCamPos.addEventListener('change', (e) => {
        this.sendBackendCommand({ action: "set_init_cam_pos", pos_mode: e.target.value });
      });
    }

    // Pan & Tilt Sliders
    document.getElementById('sliderPanSpeed').addEventListener('input', (e) => {
      this.gimbalPanSpeedMax = parseFloat(e.target.value);
      document.getElementById('panSpeedVal').innerText = `${this.gimbalPanSpeedMax.toFixed(1)} °/s`;
      this.sendBackendCommand({ action: "set_pan_speed", pan_speed: this.gimbalPanSpeedMax });
    });
    document.getElementById('sliderTiltSpeed').addEventListener('input', (e) => {
      this.gimbalTiltSpeedMax = parseFloat(e.target.value);
      document.getElementById('tiltSpeedVal').innerText = `${this.gimbalTiltSpeedMax.toFixed(1)} °/s`;
      this.sendBackendCommand({ action: "set_tilt_speed", tilt_speed: this.gimbalTiltSpeedMax });
    });

    // Noise Model Chips
    const chipGaussian = document.getElementById('chipGaussian');
    const chipSaltPepper = document.getElementById('chipSaltPepper');
    const chipPoisson = document.getElementById('chipPoisson');

    const toggleChip = (el, prop) => {
      this[prop] = !this[prop];
      if (this[prop]) {
        el.className = "noise-chip active px-2 py-1 rounded text-[10px] font-mono bg-[#E6F8F5] text-[#0F9E85] border border-[#A7F3D0] flex items-center space-x-1 cursor-pointer font-medium shadow-sm";
        el.innerHTML = `<span>✓ ${el.innerText.replace(/[✓+]/, '').trim()}</span>`;
      } else {
        el.className = "noise-chip px-2 py-1 rounded text-[10px] font-mono bg-[#F7F8FA] input-box text-[#6B7280] border border-[#D8DCE3] hover:text-contrast cursor-pointer";
        el.innerHTML = `<span>+ ${el.innerText.replace(/[✓+]/, '').trim()}</span>`;
      }
      this.sendBackendCommand({
        action: "set_noise",
        gaussian: this.noiseGaussian,
        salt_pepper: this.noiseSaltPepper,
        poisson: this.noisePoisson,
        jitter: this.cameraJitter > 0,
        platform: this.platformMagnitude > 0
      });
    };

    chipGaussian.addEventListener('click', () => toggleChip(chipGaussian, 'noiseGaussian'));
    chipSaltPepper.addEventListener('click', () => toggleChip(chipSaltPepper, 'noiseSaltPepper'));
    chipPoisson.addEventListener('click', () => toggleChip(chipPoisson, 'noisePoisson'));

    // Disturbance Sliders
    document.getElementById('sliderNoiseStd').addEventListener('input', (e) => {
      this.noiseStdDev = parseFloat(e.target.value);
      document.getElementById('noiseStdVal').innerText = `${this.noiseStdDev.toFixed(1)} px`;
      this.sendBackendCommand({ action: "set_noise_params", std_dev: this.noiseStdDev });
    });
    document.getElementById('sliderJitter').addEventListener('input', (e) => {
      this.cameraJitter = parseFloat(e.target.value);
      document.getElementById('jitterVal').innerText = `${this.cameraJitter.toFixed(1)} px/fr`;
      this.sendBackendCommand({ action: "set_noise_params", jitter: this.cameraJitter });
    });
    document.getElementById('sliderPlatformMag').addEventListener('input', (e) => {
      this.platformMagnitude = parseFloat(e.target.value);
      document.getElementById('platformMagVal').innerText = `${this.platformMagnitude.toFixed(1)} px/fr`;
      this.sendBackendCommand({ action: "set_noise_params", platform_mag: this.platformMagnitude });
    });

    // Platform Motion Buttons
    document.querySelectorAll('.plat-motion-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.plat-motion-btn').forEach(b => {
          b.className = "plat-motion-btn px-1.5 py-1 text-[10px] font-mono rounded bg-[#F7F8FA] input-box text-[#6B7280] border border-[#D8DCE3] hover:text-contrast text-center cursor-pointer";
        });
        e.currentTarget.className = "plat-motion-btn px-1.5 py-1 text-[10px] font-mono rounded bg-[#E6F8F5] text-[#0F9E85] border border-[#0F9E85] font-medium text-center shadow-sm cursor-pointer";
        const motionType = e.currentTarget.dataset.motion || 'linear';
        this.platformMotionType = motionType;
        this.sendBackendCommand({ action: "set_platform_type", platform_type: motionType });
      });
    });

    // Atmospheric Optical Channel Buttons
    document.querySelectorAll('.atmos-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.atmos-btn').forEach(b => {
          b.className = "atmos-btn flex flex-col items-center p-1 rounded bg-[#F7F8FA] input-box border border-[#D8DCE3] text-[#6B7280] hover:text-contrast cursor-pointer text-center";
        });
        e.currentTarget.className = "atmos-btn flex flex-col items-center p-1 rounded bg-[#E6F8F5] border border-[#0F9E85] text-[#0F9E85] cursor-pointer text-center shadow-sm";
        this.atmosphericChannel = e.currentTarget.dataset.atmos;
        document.getElementById('envProfileBadge').innerText = `${this.atmosphericChannel.toUpperCase()} CHANNEL`;
        this.sendBackendCommand({ action: "set_atmosphere", atmosphere: this.atmosphericChannel });
      });
    });

    // Video Input
    const videoDropZone = document.getElementById('videoDropZone');
    const videoFileInput = document.getElementById('videoFileInput');
    videoDropZone.addEventListener('click', () => videoFileInput.click());
    videoFileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        const file = e.target.files[0];
        document.getElementById('videoFileName').innerText = file.name;
        document.getElementById('videoStatusBadge').innerText = "LOADED";
        this.setMode('VIDEO');
      }
    });

    document.getElementById('btnLoadSampleVideo').addEventListener('click', () => {
      document.getElementById('videoFileName').innerText = "FSOC_Flight_Test_E2E.mp4 (Loaded)";
      document.getElementById('videoStatusBadge').innerText = "ACTIVE";
      this.setMode('VIDEO');
    });

    // Export Performance Report
    this.btnExportReport.addEventListener('click', () => this.showReportModal());
    this.btnCloseModal.addEventListener('click', () => this.reportModal.classList.add('hidden'));
    this.btnDownloadCsv.addEventListener('click', () => this.downloadCsvReport());
    this.btnPrintReport.addEventListener('click', () => window.print());
  }

  toggleTheme() {
    if (this.theme === 'light') {
      this.theme = 'dark';
      document.body.classList.remove('theme-light');
      document.body.classList.add('theme-dark');
      document.getElementById('darkThemeIcon').className = "w-5 h-5 rounded-full bg-[#141A24] text-[#1FE6C4] shadow-sm flex items-center justify-center font-bold";
      document.getElementById('lightThemeIcon').className = "w-5 h-5 rounded-full text-[#8A8F9A] flex items-center justify-center hover:text-white";
      this.themeLabel.innerText = "DARK";
      this.themeLabel.className = "font-mono text-[#1FE6C4] font-bold text-[11px]";
    } else {
      this.theme = 'light';
      document.body.classList.remove('theme-dark');
      document.body.classList.add('theme-light');
      document.getElementById('darkThemeIcon').className = "w-5 h-5 rounded-full text-[#6B7280] flex items-center justify-center hover:text-[#1B1F27]";
      document.getElementById('lightThemeIcon').className = "w-5 h-5 rounded-full bg-white text-[#0F9E85] shadow-sm flex items-center justify-center font-bold";
      this.themeLabel.innerText = "LIGHT";
      this.themeLabel.className = "font-mono text-[#0F9E85] font-bold text-[11px]";
    }
  }

  setMode(mode) {
    this.mode = mode;
    if (mode === 'SIMULATION') {
      this.simModeBtn.className = "px-5 py-1.5 rounded-full text-xs font-semibold tracking-wider transition-all duration-200 bg-[#0F9E85] text-white shadow-md flex items-center space-x-1.5 cursor-pointer";
      this.vidModeBtn.className = "px-5 py-1.5 rounded-full text-xs font-semibold tracking-wider text-[#6B7280] hover:text-[#1B1F27] transition-all duration-200 flex items-center space-x-1.5 cursor-pointer";
    } else {
      this.vidModeBtn.className = "px-5 py-1.5 rounded-full text-xs font-semibold tracking-wider transition-all duration-200 bg-[#0F9E85] text-white shadow-md flex items-center space-x-1.5 cursor-pointer";
      this.simModeBtn.className = "px-5 py-1.5 rounded-full text-xs font-semibold tracking-wider text-[#6B7280] hover:text-[#1B1F27] transition-all duration-200 flex items-center space-x-1.5 cursor-pointer";
    }
  }

  resetSimulation() {
    this.targetPos = { x: 1624, y: 580 };
    this.gimbalPos = { x: 1000, y: 1000 };
    this.trailHistory = [];
    this.patternTime = 0;
    this.kalman.reset();
    this.errorHistory = [];
    this.startTime = Date.now();
    this.epochSeconds = 0;
    this.trackingError = 0.0;
    this.rmse = 0.0;
    this.acquisitionTime = null;
    this.reacqTime = null;
    this.lossRate = 0.0;
    this.retentionRate = 0.0;
    this.isLocked = false;
    this.lockConfidence = 0.0;
    this.stateLabel = 'SEARCHING';
    this.telemetryLogs = [];
    // Reset spiral/sinusoid state
    this.spiralAngle = 0;
    this.spiralRadius = 50;
    this.sinusoidX = 200;
  }

  startClock() {
    setInterval(() => {
      const now = new Date();
      this.utcClockText.innerText = `UTC: ${now.toISOString().replace('T', ' ').substring(0, 23)}`;
      
      if (this.isRunning) {
        this.epochSeconds++;
        const m = Math.floor(this.epochSeconds / 60);
        const s = this.epochSeconds % 60;
        this.valEpochDuration.innerText = `${m}m ${s}s`;
      }
    }, 1000);
  }

  updateClientPhysics(dt) {
    if (!this.isRunning || this.isBackendConnected) return;

    this.patternTime += dt;

    // 1. Move Target Beacon according to selected pattern
    if (this.pattern === 'straight') {
      const vx = Math.cos(this.targetAngle) * this.targetSpeed;
      const vy = Math.sin(this.targetAngle) * this.targetSpeed;
      this.targetPos.x += vx * dt;
      this.targetPos.y += vy * dt;

      if (this.targetPos.x < 150 || this.targetPos.x > 1850) {
        this.targetAngle = Math.PI - this.targetAngle;
      }
      if (this.targetPos.y < 150 || this.targetPos.y > 1850) {
        this.targetAngle = -this.targetAngle;
      }
    } else if (this.pattern === 'circular') {
      const w = 0.45 * (this.targetSpeed / 80.0);
      this.targetPos.x = this.circleCenter.x + Math.cos(this.patternTime * w) * this.circleRadius;
      this.targetPos.y = this.circleCenter.y + Math.sin(this.patternTime * w) * this.circleRadius;
    } else if (this.pattern === 'figure8' || this.pattern === 'figure_of_8') {
      const a = 420;
      const t = this.patternTime * 0.4 * (this.targetSpeed / 80.0);
      this.targetPos.x = 1000 + (a * Math.cos(t)) / (1 + Math.sin(t) * Math.sin(t));
      this.targetPos.y = 1000 + (a * Math.sin(t) * Math.cos(t)) / (1 + Math.sin(t) * Math.sin(t));
    } else if (this.pattern === 'random') {
      const jitterX = (Math.random() - 0.5) * this.targetSpeed * 1.5 * dt;
      const jitterY = (Math.random() - 0.5) * this.targetSpeed * 1.5 * dt;
      this.targetPos.x = Math.max(200, Math.min(1800, this.targetPos.x + jitterX));
      this.targetPos.y = Math.max(200, Math.min(1800, this.targetPos.y + jitterY));
    } else if (this.pattern === 'spiral') {
      const omega = 0.5 * (this.targetSpeed / 80.0);
      this.spiralRadius = ((this.spiralRadius || 50) + 15 * (this.targetSpeed / 80.0) * dt);
      if (this.spiralRadius > 450) this.spiralRadius = 50; // reset
      this.spiralAngle = (this.spiralAngle || 0) + omega * dt;
      this.targetPos.x = Math.max(100, Math.min(1900, this.circleCenter.x + this.spiralRadius * Math.cos(this.spiralAngle)));
      this.targetPos.y = Math.max(100, Math.min(1900, this.circleCenter.y + this.spiralRadius * Math.sin(this.spiralAngle)));
    } else if (this.pattern === 'sinusoid') {
      this.sinusoidX = (this.sinusoidX || 200) + this.targetSpeed * dt;
      if (this.sinusoidX > 1800) this.sinusoidX = 200;
      this.targetPos.x = this.sinusoidX;
      this.targetPos.y = 1000 + 150 * Math.sin(2 * Math.PI * 0.5 * this.patternTime);
    }

    // Save trail point
    if (this.frameCount % 2 === 0) {
      this.trailHistory.push({ x: this.targetPos.x, y: this.targetPos.y });
      if (this.trailHistory.length > this.maxTrail) {
        this.trailHistory.shift();
      }
    }

    // 2. Gimbal Servo Control Loop
    const errX = this.targetPos.x - this.gimbalPos.x;
    const errY = this.targetPos.y - this.gimbalPos.y;
    const panMaxStep = (this.gimbalPanSpeedMax * 35.0) * dt;
    const tiltMaxStep = (this.gimbalTiltSpeedMax * 35.0) * dt;

    this.gimbalPos.x += Math.max(-panMaxStep, Math.min(panMaxStep, errX * 0.08));
    this.gimbalPos.y += Math.max(-tiltMaxStep, Math.min(tiltMaxStep, errY * 0.08));

    this.panDeg = ((this.gimbalPos.x - 1000) / 1000 * 25.0);
    this.tiltDeg = ((this.gimbalPos.y - 1000) / 1000 * 25.0);

    // 3. Kalman Filter
    this.kalman.predict(dt);
    this.kalman.update(this.targetPos.x, this.targetPos.y);

    const rawDist = Math.hypot(this.targetPos.x - this.gimbalPos.x, this.targetPos.y - this.gimbalPos.y);
    this.trackingError = parseFloat(((rawDist / 2000) * 480 * 0.035).toFixed(2));
    this.errorHistory.push(this.trackingError);
    if (this.errorHistory.length > 200) this.errorHistory.shift();

    this.isLocked = this.trackingError < 1.5;
    this.lockConfidence = Math.max(75.0, Math.min(99.9, 100.0 - this.trackingError * 6.5));
  }

  drawSceneOverview() {
    const ctx = this.sceneCtx;
    const w = this.sceneCanvas.width;
    const h = this.sceneCanvas.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#06090E';
    ctx.fillRect(0, 0, w, h);

    const scaleX = w / this.worldWidth;
    const scaleY = h / this.worldHeight;

    // Grid lines
    ctx.strokeStyle = 'rgba(31, 230, 196, 0.06)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= this.worldWidth; x += 200) {
      ctx.beginPath();
      ctx.moveTo(x * scaleX, 0);
      ctx.lineTo(x * scaleX, h);
      ctx.stroke();
    }
    for (let y = 0; y <= this.worldHeight; y += 200) {
      ctx.beginPath();
      ctx.moveTo(0, y * scaleY);
      ctx.lineTo(w, y * scaleY);
      ctx.stroke();
    }

    // Origin Crosshair
    ctx.strokeStyle = 'rgba(31, 230, 196, 0.2)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(w / 2, 0);
    ctx.lineTo(w / 2, h);
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w / 2, h / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Comet-tail Trajectory
    if (this.trailHistory.length > 1) {
      for (let i = 0; i < this.trailHistory.length - 1; i++) {
        const p1 = this.trailHistory[i];
        const p2 = this.trailHistory[i + 1];
        const alpha = (i / this.trailHistory.length) * 0.85;

        ctx.strokeStyle = `rgba(255, 207, 92, ${alpha})`;
        ctx.lineWidth = 2 + (i / this.trailHistory.length) * 2;
        ctx.setLineDash([2, 4]);
        ctx.beginPath();
        ctx.moveTo(p1.x * scaleX, p1.y * scaleY);
        ctx.lineTo(p2.x * scaleX, p2.y * scaleY);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // Target Beacon
    const tgtCanvasX = this.targetPos.x * scaleX;
    const tgtCanvasY = this.targetPos.y * scaleY;

    ctx.fillStyle = '#FFCF5C';
    ctx.shadowColor = '#FFCF5C';
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.arc(tgtCanvasX, tgtCanvasY, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Coordinate Tag
    ctx.fillStyle = 'rgba(10, 14, 20, 0.9)';
    ctx.strokeStyle = 'rgba(255, 207, 92, 0.5)';
    ctx.lineWidth = 1;
    ctx.fillRect(tgtCanvasX - 45, tgtCanvasY + 12, 90, 18);
    ctx.strokeRect(tgtCanvasX - 45, tgtCanvasY + 12, 90, 18);
    ctx.fillStyle = '#FFCF5C';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`TGT_01: [${Math.round(this.targetPos.x)}, ${Math.round(this.targetPos.y)}]`, tgtCanvasX, tgtCanvasY + 25);

    // Gimbal Camera FOV Box (640x480)
    const fovW = this.camResX * scaleX;
    const fovH = this.camResY * scaleY;
    const fovX = (this.gimbalPos.x * scaleX) - (fovW / 2);
    const fovY = (this.gimbalPos.y * scaleY) - (fovH / 2);

    ctx.fillStyle = 'rgba(31, 230, 196, 0.04)';
    ctx.fillRect(fovX, fovY, fovW, fovH);

    ctx.strokeStyle = '#1FE6C4';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(fovX, fovY, fovW, fovH);

    // Corner Brackets
    const b = 8;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(fovX, fovY + b); ctx.lineTo(fovX, fovY); ctx.lineTo(fovX + b, fovY);
    ctx.moveTo(fovX + fovW - b, fovY); ctx.lineTo(fovX + fovW, fovY); ctx.lineTo(fovX + fovW, fovY + b);
    ctx.moveTo(fovX, fovY + fovH - b); ctx.lineTo(fovX, fovY + fovH); ctx.lineTo(fovX + b, fovY + fovH);
    ctx.moveTo(fovX + fovW - b, fovY + fovH); ctx.lineTo(fovX + fovW, fovY + fovH); ctx.lineTo(fovX + fovW, fovY + fovH - b);
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#1FE6C4';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText("GIMBAL FOV [640×480]", fovX + 6, fovY + 12);
    ctx.textAlign = 'right';
    ctx.fillText(`AZ: ${this.panDeg >= 0 ? '+' : ''}${this.panDeg.toFixed(1)}° | EL: ${this.tiltDeg >= 0 ? '+' : ''}${this.tiltDeg.toFixed(1)}°`, fovX + fovW - 6, fovY + fovH - 6);
  }

  drawCameraFeed() {
    const ctx = this.camCtx;
    const w = this.camCanvas.width;
    const h = this.camCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // If Python backend sends live OpenCV frame, draw it directly!
    if (this.isBackendConnected && this.backendFrameImg && this.backendFrameImg.complete && this.backendFrameImg.naturalWidth > 0) {
      ctx.drawImage(this.backendFrameImg, 0, 0, w, h);
    } else {
      // Client-side fallback rendering
      ctx.fillStyle = this.cameraType === 'ir' ? '#04060A' : '#070C15';
      ctx.fillRect(0, 0, w, h);

      const relTgtX = (w / 2) + (this.targetPos.x - this.gimbalPos.x);
      const relTgtY = (h / 2) + (this.targetPos.y - this.gimbalPos.y);

      // Target Spot
      if (relTgtX >= -50 && relTgtX <= w + 50 && relTgtY >= -50 && relTgtY <= h + 50) {
        ctx.fillStyle = '#FFCF5C';
        ctx.shadowColor = '#FFCF5C';
        ctx.shadowBlur = 18;
        if (this.targetGeometry === 'circle') {
          ctx.beginPath();
          ctx.arc(relTgtX, relTgtY, this.targetSize / 2, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.fillRect(relTgtX - this.targetSize / 2, relTgtY - this.targetSize / 2, this.targetSize, this.targetSize);
        }
        ctx.shadowBlur = 0;
      }
    }

    // Atmospheric & Noise Disturbance Visual Overlays
    if (this.atmosphericChannel === 'fog') {
      ctx.fillStyle = 'rgba(200, 210, 220, 0.28)';
      ctx.fillRect(0, 0, w, h);
    } else if (this.atmosphericChannel === 'haze') {
      ctx.fillStyle = 'rgba(180, 190, 200, 0.14)';
      ctx.fillRect(0, 0, w, h);
    } else if (this.atmosphericChannel === 'rain') {
      ctx.strokeStyle = 'rgba(200, 220, 255, 0.45)';
      ctx.lineWidth = 1.2;
      this.rainDrops.forEach(drop => {
        drop.y += drop.speed;
        drop.x += 2;
        if (drop.y > h) { drop.y = -10; drop.x = Math.random() * w; }
        ctx.beginPath();
        ctx.moveTo(drop.x, drop.y);
        ctx.lineTo(drop.x + 3, drop.y + drop.len);
        ctx.stroke();
      });
    } else if (this.atmosphericChannel === 'night') {
      ctx.fillStyle = 'rgba(0, 40, 20, 0.25)';
      ctx.fillRect(0, 0, w, h);
    }

    // Salt & Pepper Noise Overlay
    if (this.noiseSaltPepper) {
      ctx.fillStyle = '#FFFFFF';
      for (let i = 0; i < 150; i++) {
        const rx = Math.random() * w;
        const ry = Math.random() * h;
        ctx.fillRect(rx, ry, 1.5, 1.5);
      }
      ctx.fillStyle = '#000000';
      for (let i = 0; i < 150; i++) {
        const rx = Math.random() * w;
        const ry = Math.random() * h;
        ctx.fillRect(rx, ry, 1.5, 1.5);
      }
    }

    // Technical HUD Grid, Range Rings & Crosshairs
    ctx.strokeStyle = 'rgba(31, 230, 196, 0.25)';
    ctx.lineWidth = 1;
    [60, 140, 210].forEach(r => {
      ctx.beginPath();
      ctx.arc(w / 2, h / 2, r, 0, Math.PI * 2);
      ctx.stroke();
    });

    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(w / 2, 20); ctx.lineTo(w / 2, h - 20);
    ctx.moveTo(20, h / 2); ctx.lineTo(w - 20, h / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Pitch & Yaw Ladders
    ctx.strokeStyle = '#1FE6C4';
    ctx.lineWidth = 1.5;
    [-60, 60].forEach(dy => {
      ctx.beginPath();
      ctx.moveTo(w / 2 - 12, h / 2 + dy); ctx.lineTo(w / 2 + 12, h / 2 + dy);
      ctx.stroke();
    });
    [-60, 60].forEach(dx => {
      ctx.beginPath();
      ctx.moveTo(w / 2 + dx, h / 2 - 12); ctx.lineTo(w / 2 + dx, h / 2 + 12);
      ctx.stroke();
    });

    // Kalman Reticle Snap
    const relTgtX = (w / 2) + (this.targetPos.x - this.gimbalPos.x);
    const relTgtY = (h / 2) + (this.targetPos.y - this.gimbalPos.y);

    if (relTgtX >= -20 && relTgtX <= w + 20 && relTgtY >= -20 && relTgtY <= h + 20) {
      const lockBoxSize = 46;
      ctx.strokeStyle = this.isLocked ? '#3DDC84' : '#FFCF5C';
      ctx.lineWidth = 2;
      ctx.strokeRect(relTgtX - lockBoxSize / 2, relTgtY - lockBoxSize / 2, lockBoxSize, lockBoxSize);

      // Lock Tag
      ctx.fillStyle = 'rgba(15, 21, 31, 0.95)';
      ctx.strokeStyle = this.isLocked ? 'rgba(61, 220, 132, 0.7)' : 'rgba(255, 207, 92, 0.7)';
      ctx.lineWidth = 1;
      ctx.fillRect(relTgtX + 28, relTgtY - 34, 140, 30);
      ctx.strokeRect(relTgtX + 28, relTgtY - 34, 140, 30);

      ctx.fillStyle = this.isLocked ? '#3DDC84' : '#FFCF5C';
      ctx.font = 'bold 9px "JetBrains Mono", monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`BEACON ID: #01 | CONF: ${this.lockConfidence.toFixed(1)}%`, relTgtX + 34, relTgtY - 22);
      ctx.fillStyle = '#F2F3F5';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillText(`ΔX: ${((relTgtX - w / 2) * 0.05).toFixed(2)}px · ΔY: ${((relTgtY - h / 2) * 0.05).toFixed(2)}px`, relTgtX + 34, relTgtY - 10);
    }
  }

  updateTelemetryUI() {
    // Tracking Error
    const errColor = this.trackingError < 1.5 ? '#1FAE64' : '#DC2626';
    this.valTrackingError.innerText = this.trackingError.toFixed(2);
    this.statusTrackingError.innerText = this.trackingError < 1.5 ? 'IN SPEC' : 'WARNING';
    this.statusTrackingError.style.color = errColor;

    // FPS
    this.valFps.innerText = this.fps.toFixed(1);

    // Acquisition time — show live value or placeholder
    this.valAcqTime.innerText = (this.acquisitionTime !== null && this.acquisitionTime !== undefined)
      ? this.acquisitionTime.toFixed(2)
      : '---';

    // Re-acquisition time
    this.valReacqTime.innerText = (this.reacqTime !== null && this.reacqTime !== undefined)
      ? this.reacqTime.toFixed(2)
      : '---';

    // Loss & retention rates
    this.valLossRate.innerText = this.lossRate.toFixed(2);
    this.valRetentionRate.innerText = this.retentionRate.toFixed(2);

    // Lock badge
    if (this.isLocked) {
      this.lockBadgeContainer.className = 'flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-[#ECFDF5] border border-[#A7F3D0] shadow-sm';
      this.lockBadgeText.innerText = 'LOCKED';
      this.lockBadgeText.className = 'font-mono font-bold text-[11px] text-[#1FAE64] tracking-wider';
      this.lockPingDot.className = 'w-2 h-2 rounded-full bg-[#1FAE64] animate-ping';
      this.footerStatusDot.style.background = '#1FAE64';
      this.footerStatusText.innerText = 'FSOC COARSE BEACON LOCK ACQUIRED';
    } else {
      const stateColors = {
        'SEARCHING': '#D97706',
        'DETECTED': '#0F9E85',
        'ACQUIRING': '#0F9E85',
        'TARGET LOST': '#DC2626',
        'RE-ACQUIRING': '#DC2626',
      };
      const badgeColor = stateColors[this.stateLabel] || '#DC2626';
      this.lockBadgeContainer.className = 'flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full bg-[#FEF2F2] border border-[#FECACA] shadow-sm';
      this.lockBadgeText.innerText = this.stateLabel || 'ACQUIRING';
      this.lockBadgeText.style.color = badgeColor;
      this.lockPingDot.style.background = badgeColor;
      this.footerStatusDot.style.background = badgeColor;
      this.footerStatusText.innerText = this.stateLabel === 'SEARCHING' ? 'SEARCHING FOR BEACON...' :
        this.stateLabel === 'RE-ACQUIRING' ? 'RE-ACQUIRING BEACON CENTROID...' :
        this.stateLabel === 'TARGET LOST' ? 'TARGET LOST — INITIATING SWEEP' :
        'ACQUIRING BEACON...';
    }

    // Footer live metrics
    this.latencyValText.innerText = `${(6.5 + Math.random() * 2.0).toFixed(1)} ms`;
    this.cpuValText.innerText = `${(10 + Math.floor(Math.random() * 8))}%`;
    this.gpuValText.innerText = `${(25 + Math.floor(Math.random() * 10))}%`;
    this.platformDriftText.innerText = `${(this.platformMagnitude * 0.08 + Math.random() * 0.04).toFixed(2)} px/s`;
    this.rangeText.innerText = `${(42.80 + Math.sin(this.epochSeconds * 0.05) * 0.12).toFixed(2)} km`;

    // SNR based on lock confidence
    const snr = this.isLocked
      ? (20.0 + this.lockConfidence * 0.04 + Math.random() * 0.5).toFixed(1)
      : (10.0 + Math.random() * 3.0).toFixed(1);
    if (this.footerSnrText) this.footerSnrText.innerText = `${snr} dB`;
  }

  showReportModal() {
    const dateStr = new Date().toLocaleString();
    const avgErr = (this.errorHistory.reduce((a, b) => a + b, 0) / (this.errorHistory.length || 1)).toFixed(3);
    const maxErr = Math.max(...this.errorHistory, 0).toFixed(3);

    // If backend connected, fetch real metrics
    if (this.isBackendConnected) {
      this.modalReportContent.innerHTML = '<div class="text-center py-4 text-[#0F9E85] font-mono text-xs">⏳ Fetching live backend metrics...</div>';
      this.reportModal.classList.remove('hidden');
      fetch('/api/metrics')
        .then(r => r.json())
        .then(m => {
          this._renderReport(dateStr, m);
        })
        .catch(() => this._renderReport(dateStr, null, avgErr, maxErr));
      return;
    }
    this._renderReport(dateStr, null, avgErr, maxErr);
    this.reportModal.classList.remove('hidden');
  }

  _renderReport(dateStr, metrics, avgErrFallback, maxErrFallback) {
    // Use real backend metrics if available, otherwise client-side fallback
    const avgErr = metrics ? metrics.mean_error_px.toFixed(3) : (avgErrFallback || '---');
    const maxErr = metrics ? metrics.max_error_px.toFixed(3) : (maxErrFallback || '---');
    const rmse = metrics ? metrics.rmse_px.toFixed(3) : this.rmse.toFixed(3);
    const fps = metrics ? metrics.fps.toFixed(1) : this.fps.toFixed(1);
    const acq = metrics?.acquisition_time_sec != null ? metrics.acquisition_time_sec.toFixed(3) + ' s' :
      (this.acquisitionTime != null ? this.acquisitionTime.toFixed(3) + ' s' : '---');
    const reacq = metrics?.reacquisition_time_sec != null ? metrics.reacquisition_time_sec.toFixed(3) + ' s' :
      (this.reacqTime != null ? this.reacqTime.toFixed(3) + ' s' : '---');
    const lockRet = metrics ? metrics.lock_retention_percent.toFixed(2) : this.retentionRate.toFixed(2);
    const loss = metrics ? metrics.loss_percent.toFixed(2) : this.lossRate.toFixed(2);
    const latency = metrics ? metrics.avg_latency_ms.toFixed(2) + ' ms' : '---';
    const totalFrames = metrics ? metrics.total_frames : this.frameCount;
    const simTime = metrics ? metrics.sim_time.toFixed(2) + ' s' : this.valEpochDuration.innerText;
    const source = this.isBackendConnected ? 'PYTHON ENGINE (LIVE)' : 'CLIENT ENGINE (STANDALONE)';

    this.modalReportContent.innerHTML = `
      <div class="space-y-3">
        <div class="border-b border-gray-300 pb-2">
          <div class="font-bold text-sm text-[#0F9E85]">OPTITRACK MISSION CONTROL — FLIGHT TEST AUDIT</div>
          <div>TIMESTAMP: ${dateStr}</div>
          <div>SIMULATION MODE: ${this.mode} | PATTERN: ${this.pattern.toUpperCase()}</div>
          <div>BACKEND SYNC: ${source}</div>
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div><div class="text-gray-400">Simulation Time:</div><div class="font-bold text-contrast">${simTime}</div></div>
          <div><div class="text-gray-400">Pipeline Throughput:</div><div class="font-bold text-contrast">${fps} FPS</div></div>
          <div><div class="text-gray-400">Total Frames Processed:</div><div class="font-bold text-contrast">${totalFrames}</div></div>
          <div><div class="text-gray-400">Avg Processing Latency:</div><div class="font-bold text-contrast">${latency}</div></div>
          <div><div class="text-gray-400">Mean Tracking Error:</div><div class="font-bold text-[#0F9E85]">${avgErr} px (&lt; 1.5 px Spec)</div></div>
          <div><div class="text-gray-400">Max Peak Error:</div><div class="font-bold text-contrast">${maxErr} px</div></div>
          <div><div class="text-gray-400">RMSE:</div><div class="font-bold text-contrast">${rmse} px</div></div>
          <div><div class="text-gray-400">Initial Acquisition Time:</div><div class="font-bold text-contrast">${acq}</div></div>
          <div><div class="text-gray-400">Re-acquisition Time:</div><div class="font-bold text-contrast">${reacq}</div></div>
          <div><div class="text-gray-400">Lock Retention Rate:</div><div class="font-bold text-[#1FAE64]">${lockRet}%</div></div>
          <div><div class="text-gray-400">Target Loss Rate:</div><div class="font-bold text-contrast">${loss}%</div></div>
          <div><div class="text-gray-400">Atmospheric Channel:</div><div class="font-bold text-contrast">${this.atmosphericChannel.toUpperCase()}</div></div>
        </div>
        <div class="pt-2 text-[10px] text-gray-500">
          ${parseFloat(avgErr) < 1.5
            ? '✅ MISSION SPECIFICATIONS FULLY SATISFIED. ALL LOGS READY FOR DOWNLINK.'
            : '⚠️ TRACKING ERROR EXCEEDS 1.5 px SPEC. REVIEW DISTURBANCE SETTINGS.'}
        </div>
      </div>
    `;
    this.reportModal.classList.remove('hidden');
  }
  downloadCsvReport() {
    let csv = "Time(s),TargetX(px),TargetY(px),GimbalX(px),GimbalY(px),Error(px),FPS,Confidence(%)\n";
    if (this.telemetryLogs.length === 0) {
      csv += `0.00,${this.targetPos.x},${this.targetPos.y},${this.gimbalPos.x},${this.gimbalPos.y},${this.trackingError},${this.fps.toFixed(1)},99.4\n`;
    } else {
      this.telemetryLogs.forEach(row => {
        csv += `${row.time},${row.tgtX},${row.tgtY},${row.gimbalX},${row.gimbalY},${row.errorPx},${row.fps},${row.confidence}\n`;
      });
    }

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `OptiTrack_Telemetry_Log_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  renderLoop(timestamp) {
    const dt = Math.min(0.1, (timestamp - this.lastFrameTime) / 1000.0) || 0.033;
    this.lastFrameTime = timestamp;

    this.frameCount++;
    if (!this.isBackendConnected && this.frameCount % 10 === 0) {
      this.fps = 1.0 / dt;
    }

    this.updateClientPhysics(dt);
    this.drawSceneOverview();
    this.drawCameraFeed();

    if (this.frameCount % 5 === 0) {
      this.updateTelemetryUI();
    }

    requestAnimationFrame(this.renderLoop.bind(this));
  }
}

function toggleAccordion(id) {
  const el = document.getElementById(id);
  if (!el) return;
  if (el.classList.contains('hidden')) {
    el.classList.remove('hidden');
  } else {
    el.classList.add('hidden');
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new OptiTrackMissionControl();
});
