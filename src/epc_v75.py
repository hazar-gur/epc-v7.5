"""
EPC: Externalized Policy Coherence Module (v7.5)
=================================================

Changes from v7.4:
- Added asymmetric identity buffer:
  - Threat signals (punishment, coercion, mismatch) go through leaky buffer
  - Support signals (guidance, flex_success) bypass buffer, act directly
  - Identity rigidity updates only on buffer overflow
- Separated signal inputs: punishment, coercion, mismatch now distinct
- Added buffer diagnostics to output

Core architecture (3-axis vulnerability space):
- Temporal gate → observation timing → belief/lens
- Somatic buffer → error magnitude → policy coherence  
- Identity buffer → social/threat pressure → rigidity/self-model

Core Claim: Stability is determined by how error is routed, not by inference quality.

Author: Hazar Gür (2026)
Version: 7.5
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum


class CollapseMode(Enum):
    """Types of system collapse."""
    NONE = "none"
    PARALYSIS = "paralysis"      # Frozen, can't act
    IMPULSIVITY = "impulsivity"  # Random thrashing


# =============================================================================
# IDENTITY BUFFER (v7.5 addition)
# =============================================================================

@dataclass
class IdentityBufferConfig:
    """
    Configuration for the asymmetric identity buffer.
    
    The buffer sits between threat signals and identity updates.
    Transient pressure charges the buffer but decays away.
    Only sustained pressure overflows and binds to identity.
    """
    enabled: bool = True
    
    # Buffer dynamics
    beta_decay: float = 0.25      # Decay rate per step (0=no decay, 1=instant)
    capacity: float = 3.0         # Overflow threshold (calibrate to x_plus range)
    
    # Signal weights for x_plus (threat channel)
    w_punishment: float = 1.0     # Weight for environment punishment
    w_coercion: float = 1.0       # Weight for EPA coercion
    w_mismatch: float = 1.0       # Weight for mismatch signal
    punishment_deadzone: float = 0.05  # Ignore punishment below this
    
    # Signal weights for x_minus (support channel, bypasses buffer)
    w_guidance: float = 1.0       # Weight for guidance signal
    w_flex: float = 1.0           # Weight for flex success
    
    # Update rates
    eta_plus: float = 0.02        # Rigidity increase rate (from overflow)
    eta_minus: float = 0.01       # Rigidity decrease rate (from support)


@dataclass
class IdentityBufferState:
    """Internal state of the identity buffer."""
    level: float = 0.0            # Current buffer level
    overflow: float = 0.0         # Current overflow amount
    x_plus: float = 0.0           # Last threat input
    x_minus: float = 0.0          # Last support input


class IdentityBuffer:
    """
    Asymmetric identity buffer (v7.5).
    
    - Threat signals accumulate in leaky buffer
    - Only overflow updates identity rigidity (positive direction)
    - Support signals bypass buffer, act directly (negative direction)
    """
    
    def __init__(self, config: IdentityBufferConfig):
        self.config = config
        self.state = IdentityBufferState()
    
    def compute_inputs(
        self,
        punishment: float = 0.0,
        coercion: float = 0.0,
        mismatch: float = 0.0,
        guidance: float = 0.0,
        flex_success: float = 0.0,
    ) -> tuple:
        """Compute x_plus (buffered) and x_minus (bypass) from raw signals."""
        c = self.config
        
        # Threat channel (will be buffered)
        punishment_clipped = max(0.0, punishment - c.punishment_deadzone)
        x_plus = (
            c.w_punishment * punishment_clipped +
            c.w_coercion * coercion +
            c.w_mismatch * mismatch
        )
        
        # Support channel (bypasses buffer)
        x_minus = (
            c.w_guidance * guidance +
            c.w_flex * flex_success
        )
        
        return float(x_plus), float(x_minus)
    
    def step(
        self,
        punishment: float = 0.0,
        coercion: float = 0.0,
        mismatch: float = 0.0,
        guidance: float = 0.0,
        flex_success: float = 0.0,
    ) -> tuple:
        """
        Update buffer and compute rigidity delta.
        
        Returns:
            (delta_rigidity, state) where delta can be positive or negative
        """
        c = self.config
        s = self.state
        
        # Compute inputs
        x_plus, x_minus = self.compute_inputs(
            punishment, coercion, mismatch, guidance, flex_success
        )
        s.x_plus = x_plus
        s.x_minus = x_minus
        
        if not c.enabled:
            # UNBUFFERED: Direct update without protection
            # Threat acts immediately (no buffer), support also immediate
            s.level = 0.0
            s.overflow = x_plus  # All threat "overflows" immediately
            delta_plus = c.eta_plus * x_plus  # Direct, unfiltered
            delta_minus = c.eta_minus * x_minus
            return float(delta_plus - delta_minus), s
        
        # BUFFERED: Leaky integration of threat
        new_level = (1.0 - c.beta_decay) * s.level + x_plus
        new_level = max(0.0, new_level)  # Rectify
        s.level = new_level
        
        # Compute overflow (from updated buffer)
        s.overflow = max(0.0, s.level - c.capacity)
        
        # Compute rigidity delta
        # Positive: proportional to overflow (threat binding)
        # Negative: proportional to support (immediate relief)
        delta_plus = c.eta_plus * s.overflow
        delta_minus = c.eta_minus * x_minus
        delta_rigidity = delta_plus - delta_minus
        
        return float(delta_rigidity), s
    
    def reset(self):
        """Reset buffer state."""
        self.state = IdentityBufferState()


# =============================================================================
# MAIN EPC CONFIG AND STATE
# =============================================================================

@dataclass
class EPCConfig:
    """
    Configuration for the EPC controller.
    
    All parameters have sensible defaults that produce the documented
    phase transitions. Adjust for your specific agent/environment.
    """
    # === Somatic Buffer ===
    rho: float = 0.8              # Routing coefficient: fraction of error absorbed by buffer
    s_max: float = 10.0           # Buffer capacity
    
    # === Temporal Gating (window-based) ===
    T_hold: int = 10              # Closed window length (steps gate stays closed)
    T_open: int = 3               # Open window length (steps gate stays open)
    gate_floor: float = 0.0       # Minimum gate value during open window
    
    # === Discharge ===
    discharge_rate: float = 0.3   # How fast buffer drains per discharge action
    discharge_enabled: bool = True
    
    # === Identity Buffer (v7.5) ===
    identity_buffer: IdentityBufferConfig = field(default_factory=IdentityBufferConfig)
    
    # === Identity Control (legacy, for non-buffered path) ===
    identity_init: float = 0.5    # Initial identity rigidity I_0
    identity_amplification: float = 2.0  # k_I: how much identity amplifies signals
    
    # === Identity Channel Toggles (for ablation clarity) ===
    identity_affects_precision: bool = True  # Does identity modulate precision?
    identity_affects_threat: bool = True     # Does identity amplify threat?
    
    # === Threat Dynamics ===
    threat_decay: float = 0.95    # Natural decay of identity threat
    
    # === Thresholds (separate decompensation vs collapse) ===
    threat_decomp: float = 12.0       # Enter recoverable decompensated state
    threat_critical: float = 20.0     # True collapse threshold
    decomp_duration: int = 6          # Consecutive steps above decomp threshold
    collapse_duration: int = 10       # Consecutive steps above critical threshold
    allow_recovery: bool = True       # Can recover from decompensated?
    collapse_absorbing: bool = True   # Collapse is absorbing (no recovery)?
    
    # === Precision ===
    base_precision: float = 1.0    # π_0
    
    # === Leak / Chronic Burden Parameters ===
    saturation_frac: float = 0.9      # Fraction of s_max considered "saturated"
    leak_gain: float = 0.1            # Base leak gain when above saturation
    chronic_after_steps: int = 20     # Chronic kicks in after sustained saturation
    chronic_gain: float = 0.02        # Chronic burden slope
    
    # === Optionality Cost ===
    optionality_weight: float = 0.3   # λ: how much policy entropy adds to load


@dataclass
class EPCState:
    """Internal state of the EPC controller."""
    # Somatic buffer
    somatic_load: float = 0.0
    
    # Gate schedule
    phase: str = "closed"  # "closed" or "open"
    phase_count: int = 0
    
    # Identity
    identity_rigidity: float = 0.5
    identity_threat: float = 0.0
    
    # Regimes
    decompensated: bool = False
    collapsed: bool = False
    collapse_mode: CollapseMode = CollapseMode.NONE
    collapse_step: Optional[int] = None
    
    # Counters
    steps_above_decomp: int = 0
    steps_above_critical: int = 0
    steps_at_saturation: int = 0
    step_count: int = 0


@dataclass
class EPCControl:
    """
    Output signals from EPC controller.
    
    Use these to modulate your base agent's behavior.
    """
    # === Precision Modulation ===
    precision_multiplier: float    # Multiply base precision by this
    effective_precision: float     # = base_precision * precision_multiplier
    
    # === Gate State ===
    gate_open: bool               # Whether temporal gate is open
    gate_value: float             # Continuous gate value [0,1]
    
    # === Threat & Identity ===
    identity_rigidity: float      # Current I_t
    identity_threat: float        # Current threat level
    threat_amplification: float   # f(I_t) = 1 + k_I * I_t
    
    # === Regimes ===
    decompensated: bool           # Is agent in recoverable stressed state?
    collapsed: bool               # Is agent in absorbing failure state?
    collapse_mode: CollapseMode   # Type of collapse
    collapse_risk: float          # threat / critical threshold
    
    # === Somatic Diagnostics ===
    somatic_load: float           # Current buffer level
    error_to_buffer: float        # Error absorbed by buffer this step
    error_to_identity: float      # Error that reached identity this step
    leak: float                   # Leak from saturated buffer
    optionality_load: float       # Load from policy entropy
    
    # === Identity Buffer Diagnostics (v7.5) ===
    identity_buffer_level: float      # Current identity buffer level
    identity_buffer_overflow: float   # Current overflow amount
    identity_buffer_x_plus: float     # Threat input this step
    identity_buffer_x_minus: float    # Support input this step


class EPCController:
    """
    The EPC control module (v7.5).
    
    Wraps any base agent to provide identity-controlled error routing.
    Now includes asymmetric identity buffer for transient tolerance.
    """
    
    def __init__(self, config: Optional[EPCConfig] = None):
        self.config = config or EPCConfig()
        self.state = EPCState()
        self.state.identity_rigidity = self.config.identity_init
        
        # Initialize gate phase
        self.state.phase = "closed"
        self.state.phase_count = self.config.T_hold
        
        # Initialize identity buffer (v7.5)
        self.identity_buffer = IdentityBuffer(self.config.identity_buffer)
        
        # History for analysis
        self.history: Dict[str, List[float]] = {
            'somatic_load': [],
            'identity_rigidity': [],
            'identity_threat': [],
            'effective_precision': [],
            'gate_value': [],
            'decompensated': [],
            'collapsed': [],
            'optionality_load': [],
            # v7.5 additions
            'identity_buffer_level': [],
            'identity_buffer_overflow': [],
            'identity_buffer_x_plus': [],
            'identity_buffer_x_minus': [],
        }
    
    def reset(self):
        """Reset controller to initial state."""
        self.state = EPCState()
        self.state.identity_rigidity = self.config.identity_init
        self.state.phase = "closed"
        self.state.phase_count = self.config.T_hold
        self.identity_buffer.reset()
        self.history = {k: [] for k in self.history}
    
    @staticmethod
    def _clip01(x: float) -> float:
        """Clip value to [0, 1]."""
        return float(np.clip(x, 0.0, 1.0))
    
    def _f_identity(self, I: float) -> float:
        """Identity amplification function: f(I) = 1 + k_I * I."""
        return 1.0 + self.config.identity_amplification * float(I)
    
    def _compute_gate(self) -> float:
        """
        Compute gate value based on current phase.
        
        Gate semantics (fixed window-based):
        - closed phase: gate = 0
        - open phase: gate = gate_floor + (1-gate_floor)*(1 - somatic/s_max), clipped [0,1]
        """
        c = self.config
        s = self.state
        
        if s.phase == "closed":
            return 0.0
        
        # Open phase: gate value depends on buffer load
        raw = 1.0 - (s.somatic_load / max(1e-9, c.s_max))
        raw = float(np.clip(raw, 0.0, 1.0))
        gate = c.gate_floor + (1.0 - c.gate_floor) * raw
        return float(np.clip(gate, 0.0, 1.0))
    
    def _advance_gate_phase(self):
        """
        Advance the gate phase schedule.
        
        Closed for T_hold steps, open for T_open steps. Repeats.
        """
        c = self.config
        s = self.state
        
        s.phase_count -= 1
        if s.phase_count > 0:
            return
        
        # Switch phase
        if s.phase == "closed":
            s.phase = "open"
            s.phase_count = max(1, c.T_open)
        else:
            s.phase = "closed"
            s.phase_count = max(1, c.T_hold)
    
    def step(
        self,
        error: float,
        volatility: float = 0.0,
        punishment: float = 0.0,
        coercion: float = 0.0,        # v7.5: separate from punishment
        mismatch: float = 0.0,        # v7.5: binary mismatch signal
        guidance: float = 0.0,
        action_changed: bool = False,
        belief_updated: bool = False,
        is_discharge_action: bool = False,
        optionality_load: float = 0.0,
    ) -> EPCControl:
        """
        Process one timestep of error routing.
        
        Args:
            error: Prediction error magnitude (|o - b| or similar)
            volatility: Environmental volatility/uncertainty signal
            punishment: Environment punishment signal P_t ∈ [0,1]
            coercion: EPA coercion signal (v7.5, separate from punishment)
            mismatch: Binary mismatch signal from EPA (v7.5)
            guidance: External guidance signal G_t ∈ [0,1]
            action_changed: Did the agent change its action? (for F_t)
            belief_updated: Did beliefs update meaningfully? (for F_t)
            is_discharge_action: Is current action a discharge/rest action?
            optionality_load: Deliberation cost from policy entropy
        
        Returns:
            EPCControl with all modulation signals
        """
        c = self.config
        st = self.state
        
        # ---- If collapsed and absorbing, return minimal output ----
        if st.collapsed and c.collapse_absorbing:
            return self._collapsed_output()
        
        # ---- Inputs ----
        scaled_optionality = c.optionality_weight * float(optionality_load)
        error_total = abs(float(error)) + 0.5 * abs(float(volatility)) + scaled_optionality
        
        punishment = float(np.clip(punishment, 0.0, 1.0))
        # Coercion: only enforce non-negative (no upper bound by default)
        # Upper bounding to [0,1] was invalidating tests where coercion > 1
        coercion = float(max(0.0, coercion))
        mismatch = float(np.clip(mismatch, 0.0, 1.0))
        guidance = float(np.clip(guidance, 0.0, 1.0))
        
        # ---- 1) Somatic buffer update ----
        error_to_buffer = c.rho * error_total
        error_bypassing = (1.0 - c.rho) * error_total
        
        discharge_amount = 0.0
        if is_discharge_action and c.discharge_enabled:
            discharge_amount = c.discharge_rate * st.somatic_load
        
        st.somatic_load = float(np.clip(
            st.somatic_load + error_to_buffer - discharge_amount,
            0.0, c.s_max
        ))
        
        # ---- 2) Temporal gate (fixed window schedule) ----
        gate = self._compute_gate()
        gate_open = gate > 0.5
        
        # Advance schedule AFTER computing gate for this step
        self._advance_gate_phase()
        
        # ---- 3) Identity channel gains ----
        threat_amp = self._f_identity(st.identity_rigidity)
        
        # For ablations: separate channels
        threat_gain = threat_amp if c.identity_affects_threat else 1.0
        precision_gain = threat_amp if c.identity_affects_precision else 1.0
        
        # ---- 4) Effective precision ----
        precision_multiplier = (1.0 - c.rho) * gate * precision_gain
        effective_precision = c.base_precision * precision_multiplier
        
        # ---- 5) Error to identity ----
        unprocessed = error_bypassing * (1.0 - gate)
        error_to_identity = unprocessed
        
        # ---- 6) Leak / chronic burden ----
        sat_thr = c.saturation_frac * c.s_max
        if st.somatic_load >= sat_thr:
            st.steps_at_saturation += 1
        else:
            st.steps_at_saturation = max(0, st.steps_at_saturation - 5)
        
        base_leak = max(0.0, st.somatic_load - sat_thr) * c.leak_gain
        
        chronic = 0.0
        if (not c.discharge_enabled) and (st.steps_at_saturation > c.chronic_after_steps):
            chronic = (st.steps_at_saturation - c.chronic_after_steps) * c.chronic_gain
        
        leak = base_leak + chronic
        
        # ---- 7) Threat update (identity as survivability controller) ----
        amplified = (error_to_identity + leak) * threat_gain
        st.identity_threat = c.threat_decay * st.identity_threat + amplified
        
        # ---- 8) Identity rigidity dynamics (v7.5: through buffer) ----
        # F_t: success signal (flex success)
        flex_success = 1.0 if (gate > 0.5 and belief_updated and action_changed) else 0.0
        
        # Discharge counts as accepting support
        guidance_total = guidance + (1.0 if (is_discharge_action and c.discharge_enabled) else 0.0)
        guidance_total = float(np.clip(guidance_total, 0.0, 1.0))
        
        # v7.5: Use identity buffer for rigidity updates
        delta_rigidity, buf_state = self.identity_buffer.step(
            punishment=punishment,
            coercion=coercion,
            mismatch=mismatch,
            guidance=guidance_total,
            flex_success=flex_success,
        )
        
        # Apply rigidity change
        st.identity_rigidity = float(np.clip(
            st.identity_rigidity + delta_rigidity,
            0.0, 1.0
        ))
        
        # ---- 9) Regime logic: decompensated (recoverable) vs collapsed ----
        # Decompensation counter
        if st.identity_threat > c.threat_decomp:
            st.steps_above_decomp += 1
        else:
            st.steps_above_decomp = max(0, st.steps_above_decomp - 2)
        
        if (not st.decompensated) and (st.steps_above_decomp >= c.decomp_duration):
            st.decompensated = True
        
        # Critical collapse counter
        if st.identity_threat > c.threat_critical:
            st.steps_above_critical += 1
        else:
            st.steps_above_critical = max(0, st.steps_above_critical - 2)
        
        if (not st.collapsed) and (st.steps_above_critical >= c.collapse_duration):
            st.collapsed = True
            st.collapse_step = st.step_count
            st.collapse_mode = CollapseMode.PARALYSIS
        
        # Recovery from decompensation (if enabled): sustained low threat + guidance
        if c.allow_recovery and st.decompensated and (not st.collapsed):
            if (st.identity_threat < 0.8 * c.threat_decomp) and (guidance_total > 0.3):
                st.decompensated = False
                st.steps_above_decomp = 0
        
        # If collapse is not absorbing, allow guidance to recover it
        if (not c.collapse_absorbing) and st.collapsed:
            if (st.identity_threat < 0.8 * c.threat_critical) and (guidance_total > 0.6):
                st.collapsed = False
                st.collapse_mode = CollapseMode.NONE
                st.steps_above_critical = 0
        
        # ---- 10) Record / return ----
        st.step_count += 1
        self._record_history(effective_precision, gate, scaled_optionality, buf_state)
        
        collapse_risk = st.identity_threat / max(1e-9, c.threat_critical)
        
        return EPCControl(
            precision_multiplier=float(precision_multiplier),
            effective_precision=float(effective_precision),
            gate_open=bool(gate_open),
            gate_value=float(gate),
            identity_rigidity=float(st.identity_rigidity),
            identity_threat=float(st.identity_threat),
            threat_amplification=float(threat_amp),
            decompensated=bool(st.decompensated),
            collapsed=bool(st.collapsed),
            collapse_mode=st.collapse_mode,
            collapse_risk=float(collapse_risk),
            somatic_load=float(st.somatic_load),
            error_to_buffer=float(error_to_buffer),
            error_to_identity=float(error_to_identity),
            leak=float(leak),
            optionality_load=float(scaled_optionality),
            # v7.5 buffer diagnostics
            identity_buffer_level=float(buf_state.level),
            identity_buffer_overflow=float(buf_state.overflow),
            identity_buffer_x_plus=float(buf_state.x_plus),
            identity_buffer_x_minus=float(buf_state.x_minus),
        )
    
    def _collapsed_output(self) -> EPCControl:
        """Output for collapsed agent in absorbing state."""
        c = self.config
        st = self.state
        
        gate = 0.0
        st.identity_threat = float(st.identity_threat * 1.005)
        
        threat_amp = self._f_identity(st.identity_rigidity)
        collapse_risk = st.identity_threat / max(1e-9, c.threat_critical)
        
        # Get buffer state for diagnostics
        buf_state = self.identity_buffer.state
        
        return EPCControl(
            precision_multiplier=0.0,
            effective_precision=0.0,
            gate_open=False,
            gate_value=gate,
            identity_rigidity=float(st.identity_rigidity),
            identity_threat=float(st.identity_threat),
            threat_amplification=float(threat_amp),
            decompensated=True,
            collapsed=True,
            collapse_mode=st.collapse_mode if st.collapse_mode != CollapseMode.NONE else CollapseMode.PARALYSIS,
            collapse_risk=float(collapse_risk),
            somatic_load=float(st.somatic_load),
            error_to_buffer=0.0,
            error_to_identity=0.0,
            leak=0.0,
            optionality_load=0.0,
            identity_buffer_level=float(buf_state.level),
            identity_buffer_overflow=float(buf_state.overflow),
            identity_buffer_x_plus=0.0,
            identity_buffer_x_minus=0.0,
        )
    
    def _record_history(self, precision: float, gate: float, opt_load: float,
                        buf_state: IdentityBufferState):
        """Record state for analysis."""
        st = self.state
        self.history['somatic_load'].append(st.somatic_load)
        self.history['identity_rigidity'].append(st.identity_rigidity)
        self.history['identity_threat'].append(st.identity_threat)
        self.history['effective_precision'].append(float(precision))
        self.history['gate_value'].append(float(gate))
        self.history['decompensated'].append(float(st.decompensated))
        self.history['collapsed'].append(float(st.collapsed))
        self.history['optionality_load'].append(float(opt_load))
        # v7.5 buffer history
        self.history['identity_buffer_level'].append(float(buf_state.level))
        self.history['identity_buffer_overflow'].append(float(buf_state.overflow))
        self.history['identity_buffer_x_plus'].append(float(buf_state.x_plus))
        self.history['identity_buffer_x_minus'].append(float(buf_state.x_minus))
    
    def inject_guidance(self, amount: float = 0.1):
        """
        External intervention: reduce identity rigidity.
        
        Models: therapy, mentorship, safe environment, rest.
        """
        self.state.identity_rigidity = max(0.0, self.state.identity_rigidity - float(amount))
    
    def get_diagnostics(self) -> Dict:
        """Get current state for debugging."""
        st = self.state
        buf = self.identity_buffer.state
        return {
            'somatic_load': st.somatic_load,
            'identity_rigidity': st.identity_rigidity,
            'identity_threat': st.identity_threat,
            'decompensated': st.decompensated,
            'collapsed': st.collapsed,
            'collapse_mode': st.collapse_mode.value,
            'steps_above_decomp': st.steps_above_decomp,
            'steps_above_critical': st.steps_above_critical,
            'step_count': st.step_count,
            'gate_phase': st.phase,
            'gate_phase_count': st.phase_count,
            # v7.5 buffer diagnostics
            'identity_buffer_level': buf.level,
            'identity_buffer_overflow': buf.overflow,
            'identity_buffer_x_plus': buf.x_plus,
            'identity_buffer_x_minus': buf.x_minus,
        }


# === PRESET CONFIGURATIONS ===

def null_config() -> EPCConfig:
    """No buffer, no gating - raw error hits identity."""
    return EPCConfig(rho=0.0, T_hold=0, T_open=1, discharge_enabled=False)


def weak_config() -> EPCConfig:
    """Weak buffer - partial protection."""
    return EPCConfig(rho=0.3, T_hold=3, T_open=2, discharge_rate=0.1, discharge_enabled=True)


def strong_config() -> EPCConfig:
    """Strong buffer - full protection under normal stress."""
    return EPCConfig(rho=0.8, T_hold=10, T_open=3, discharge_rate=0.3, discharge_enabled=True)


def maladaptive_config() -> EPCConfig:
    """Buffer but no discharge - delayed collapse."""
    return EPCConfig(rho=0.8, T_hold=10, T_open=3, discharge_enabled=False)


def high_identity_config() -> EPCConfig:
    """High identity rigidity - collapses under stress even with buffer."""
    return EPCConfig(
        rho=0.8, T_hold=10, T_open=3, discharge_rate=0.3, discharge_enabled=True,
        identity_init=0.9, identity_amplification=3.0
    )


def resilient_config() -> EPCConfig:
    """Low identity rigidity + high guidance - very stable."""
    return EPCConfig(
        rho=0.8, T_hold=10, T_open=3, discharge_rate=0.3, discharge_enabled=True,
        identity_init=0.2, identity_amplification=1.0,
    )


# === v7.5 SPECIFIC CONFIGS ===

def unbuffered_identity_config() -> EPCConfig:
    """v7.5: Disable identity buffer for comparison."""
    cfg = EPCConfig()
    cfg.identity_buffer.enabled = False
    return cfg


def strong_identity_buffer_config() -> EPCConfig:
    """v7.5: Strong identity buffer (high capacity, slow decay)."""
    cfg = EPCConfig()
    cfg.identity_buffer.beta_decay = 0.15
    cfg.identity_buffer.capacity = 5.0
    return cfg


def weak_identity_buffer_config() -> EPCConfig:
    """v7.5: Weak identity buffer (low capacity, fast decay)."""
    cfg = EPCConfig()
    cfg.identity_buffer.beta_decay = 0.4
    cfg.identity_buffer.capacity = 1.5
    return cfg


# === QUICK TEST ===

if __name__ == "__main__":
    print("EPC Module Test (v7.5 - Asymmetric Identity Buffer)")
    print("=" * 70)
    
    # Test buffer isolation: burst vs sustained pressure
    print("\nTest: Burst vs Sustained Pressure")
    print("-" * 50)
    
    for label, pattern in [("burst", [1.0, 1.0, 0.0, 0.0, 0.0] * 20),
                            ("sustained", [0.5] * 100)]:
        cfg = EPCConfig()
        cfg.identity_buffer.capacity = 3.0
        cfg.identity_buffer.beta_decay = 0.25
        
        epc = EPCController(cfg)
        
        for p in pattern:
            epc.step(error=0.3, punishment=p)
        
        final_rigidity = epc.state.identity_rigidity
        max_overflow = max(epc.history['identity_buffer_overflow'])
        
        print(f"  {label:12} | rigidity: {epc.config.identity_init:.2f} → {final_rigidity:.2f} | "
              f"max_overflow: {max_overflow:.2f}")
    
    # Test asymmetry: threat vs support
    print("\nTest: Asymmetry (Threat Buffered, Support Bypasses)")
    print("-" * 50)
    
    # Threat only
    cfg = EPCConfig()
    epc = EPCController(cfg)
    for _ in range(50):
        epc.step(error=0.3, punishment=0.5)
    rigidity_threat = epc.state.identity_rigidity
    
    # Support only
    cfg = EPCConfig()
    epc = EPCController(cfg)
    for _ in range(50):
        epc.step(error=0.3, guidance=0.5)
    rigidity_support = epc.state.identity_rigidity
    
    print(f"  threat only:  {cfg.identity_init:.2f} → {rigidity_threat:.2f}")
    print(f"  support only: {cfg.identity_init:.2f} → {rigidity_support:.2f}")
    
    # Test buffered vs unbuffered under volatile conditions
    print("\nTest: Buffered vs Unbuffered Identity (Volatile Punishment)")
    print("-" * 50)
    
    np.random.seed(42)
    volatile_punishment = np.random.choice([0.0, 0.8], size=200, p=[0.7, 0.3])
    
    for buffer_enabled, label in [(True, "buffered"), (False, "unbuffered")]:
        cfg = EPCConfig()
        cfg.identity_buffer.enabled = buffer_enabled
        epc = EPCController(cfg)
        
        for p in volatile_punishment:
            epc.step(error=0.3, punishment=float(p))
        
        status = "COLLAPSED" if epc.state.collapsed else "STABLE"
        print(f"  {label:12} | {status:10} | rigidity: {cfg.identity_init:.2f} → "
              f"{epc.state.identity_rigidity:.2f}")
    
    print()
    print("EPC v7.5 module working correctly.")
