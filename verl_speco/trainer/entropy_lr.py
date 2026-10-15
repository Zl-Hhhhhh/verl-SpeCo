"""Bounded entropy LR control and paired rollout evidence (no device work)."""
from dataclasses import asdict, dataclass
import math
import statistics


def trace_statistics(traces, threshold=0.01):
    reached = low = rejected_low = accepted_total = events_total = 0
    per_prompt = []
    for events in traces:
        if events is None:
            raise ValueError("Entropy LR requires verification traces for every response")
        accepted_sum = event_count = 0
        for event in events:
            n, a = int(event['draft_count']), int(event['accepted_count'])
            if not 0 <= a <= n or n <= 0:
                raise ValueError("Invalid verification counts")
            count = min(n, a + 1)
            h = event.get('draft_teacher_entropies', [])
            if len(h) < count:
                raise ValueError("Entropy LR requires reached-position entropy")
            for i, value in enumerate(h[:count]):
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid verification entropy")
                reached += 1
                if value < threshold:
                    low += 1
                    rejected_low += int(i == a and a < n)
            accepted_sum += a
            event_count += 1
        if not event_count:
            raise ValueError("Empty response trace cannot support paired validation")
        per_prompt.append(1.0 + accepted_sum / event_count)
        accepted_total += accepted_sum
        events_total += event_count
    if not reached:
        raise ValueError("Entropy LR received no reached positions")
    return dict(low_entropy_share=low / reached, mismatch=rejected_low / low if low else 0.0,
                reached=reached, low=low, rejected_low=rejected_low,
                acceptance=1.0 + accepted_total / events_total, per_prompt=per_prompt)


def paired_evidence(
    baseline, candidate, *, projected_tokens, overhead_s, degradation_threshold=0.05
):
    a, b = baseline['per_prompt'], candidate['per_prompt']
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Paired validation needs equal cohorts of at least two prompts")
    for result in (baseline, candidate):
        values = [*result['per_prompt'], result['acceptance'], result['mismatch'], result['ms_per_token']]
        if not all(math.isfinite(value) for value in values) or result['ms_per_token'] <= 0:
            raise ValueError("Paired validation requires finite measurements and positive latency")
    if not all(math.isfinite(value) and value >= 0 for value in (projected_tokens, overhead_s)):
        raise ValueError("Invalid validation cost estimate")
    differences = [y - x for x, y in zip(a, b, strict=True)]
    mean = statistics.mean(differences)
    # Conservative screening interval; this is not a formal repeated-testing claim.
    margin = 2.1 * statistics.stdev(differences) / math.sqrt(len(differences))
    gain = candidate['acceptance'] - baseline['acceptance']
    mismatch_delta = candidate['mismatch'] - baseline['mismatch']
    net_s = projected_tokens * (baseline['ms_per_token'] - candidate['ms_per_token']) / 1000 - overhead_s
    latency_ratio = candidate['ms_per_token'] / max(baseline['ms_per_token'], 1e-12)
    degraded = mean + margin < -degradation_threshold or latency_ratio > 1.15
    improved = mean - margin > 0 and mismatch_delta < 0 and net_s > 0
    return dict(improved=int(improved), degraded=int(degraded), acceptance_delta=gain,
                paired_mean=mean, paired_lower=mean-margin, paired_upper=mean+margin,
                mismatch_delta=mismatch_delta, estimated_net_seconds=net_s,
                latency_ratio=latency_ratio, prompts=len(a))


@dataclass
class EntropyLRController:
    base: float = 1e-5
    floor: float = 5e-6
    maximum: float = 1.5e-5
    ema_beta: float = 0.3
    current: float = 1e-5
    certainty_ema: float | None = None
    mismatch: float = 0.0
    good_streak: int = 0
    saturation: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.floor, self.base, self.maximum, self.current, self.ema_beta)):
            raise ValueError("Entropy LR settings must be finite")
        if not 0 < self.floor <= self.base <= self.maximum or not 0 < self.ema_beta <= 1:
            raise ValueError("Invalid entropy LR range or EMA beta")

    @property
    def ceiling(self):
        c = self.certainty_ema if self.certainty_ema is not None else 0.0
        return self.base + (self.maximum-self.base) * min(1.0, max(0.0, c))

    def observe(self, stats):
        c, m = float(stats['low_entropy_share']), float(stats['mismatch'])
        if not 0 <= c <= 1 or not 0 <= m <= 1:
            raise ValueError("Invalid certainty or mismatch")
        self.certainty_ema = c if self.certainty_ema is None else self.ema_beta*c + (1-self.ema_beta)*self.certainty_ema
        self.mismatch = m
        self.current = max(self.floor, min(self.current, self.ceiling))
        return self.current

    def feedback(self, *, improved=False, degraded=False, saturation=0.0):
        self.saturation = float(saturation)
        if degraded:
            self.current = max(self.floor, self.current * .5)
            self.good_streak = 0
        elif improved and self.mismatch > .10:
            self.good_streak += 1
            self.current = min(self.ceiling, self.current * 1.1)
        else:
            self.good_streak = 0
        return self.current

    def metrics(self):
        return {f'drafter/entropy_lr/{k}': v for k, v in {
            'next_lr': self.current, 'ceiling': self.ceiling,
            'certainty_ema': self.certainty_ema or 0.0, 'mismatch': self.mismatch,
            'good_streak': self.good_streak, 'saturation': self.saturation,
        }.items()}

    def state_dict(self):
        return asdict(self)


def set_cycle_lr(optimizer, scheduler, lr):
    """Keep constant scheduler steps from overwriting this cycle's selected LR."""
    lr = float(lr)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Cycle LR must be finite and positive")
    for group in optimizer.param_groups:
        group['lr'] = group['initial_lr'] = lr
    if scheduler is not None:
        scheduler.base_lrs = [lr] * len(optimizer.param_groups)
        scheduler._last_lr = [lr] * len(optimizer.param_groups)
