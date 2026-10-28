package app.presence.engine

/**
 * One scored speech observation. [timestampMs] is a position in the audio stream.
 *
 * @param embedding the voice this observation scored, carried so that a detection can hand the
 *   parent's "not them" correction the exact embeddings that caused it. Optional because a replay
 *   from cached scores has no embeddings and needs none.
 */
data class Observation(
    val timestampMs: Long,
    val logLikelihoodRatio: Float,
    val embedding: Embedding? = null,
)

/**
 * A threshold crossing.
 *
 * @param evidence summed log-likelihood ratio at the moment of firing
 * @param contributors the observations in the window that summed to [evidence], oldest first
 */
data class Firing(
    val evidence: Float,
    val contributors: List<Observation>,
)

/**
 * @param windowMs how far back evidence counts. Longer is more accurate and slower to react;
 *   this trade-off is the core tuning decision of the product (spec §9).
 * @param fireThreshold accumulated evidence required to trigger. The real operating point —
 *   Phase 0 sweeps it rather than guessing.
 * @param cooldownMs quiet period after firing, so one long conversation produces one
 *   intervention rather than a stream of them. 45 s matches the design handoff (`tokens.json`
 *   `cooldown.afterFire`). Because evidence is not gathered while suppressed, the real gap before
 *   a second intervention is this plus a fresh accumulation.
 */
data class AccumulatorConfig(
    val windowMs: Long = 20_000,
    val fireThreshold: Float = 6.0f,
    val cooldownMs: Long = 45_000,
) {
    init {
        require(windowMs > 0) { "windowMs must be positive" }
        require(cooldownMs >= 0) { "cooldownMs must not be negative" }
    }
}

/**
 * Sums per-segment evidence over a rolling window and fires when it crosses the threshold.
 *
 * Rung L2, and **the primary accuracy mechanism of this design** (spec §4.1) — not a debouncer.
 * Per-utterance error on children's spontaneous speech is far too high to decide on one
 * 0.9-second word; a child seeking attention talks for tens of seconds, and accumulating weak
 * evidence across that span is what makes a confident decision possible.
 */
class EvidenceAccumulator(
    private val config: AccumulatorConfig = AccumulatorConfig(),
) {
    private val window = ArrayDeque<Observation>()
    private var cooldownStartMs: Long? = null
    private var lastTimestampMs: Long? = null

    /** Returns the firing if this observation triggers a detection, else null. */
    fun accept(observation: Observation): Firing? {
        val previous = lastTimestampMs
        require(previous == null || observation.timestampMs >= previous) {
            "observations must arrive in non-decreasing timestamp order: " +
                "got ${observation.timestampMs} after $previous"
        }
        lastTimestampMs = observation.timestampMs

        val suppressedFrom = cooldownStartMs
        if (suppressedFrom != null && observation.timestampMs - suppressedFrom < config.cooldownMs) {
            // Deliberately does not accumulate. Evidence gathered while suppressed would
            // otherwise fire the instant the cooldown lapses, with no new speech since — which
            // makes the cooldown a delay rather than a requirement for fresh reason to interrupt.
            return null
        }

        window.addLast(observation)
        // Strict `<`: an observation exactly windowMs old still counts.
        val cutoff = observation.timestampMs - config.windowMs
        while (window.isNotEmpty() && window.first().timestampMs < cutoff) {
            window.removeFirst()
        }

        var total = 0f
        for (entry in window) total += entry.logLikelihoodRatio
        if (total < config.fireThreshold) return null

        cooldownStartMs = observation.timestampMs
        val contributors = window.toList()
        window.clear()
        return Firing(total, contributors)
    }

    /**
     * Starts a fresh cooldown at [atMs] and discards accumulated evidence.
     *
     * Called when the parent dismisses an intervention, so that dismissing does not let the app
     * reappear the moment a cooldown measured from the original fire runs out. [atMs] is a
     * position in the same audio stream as [Observation.timestampMs]; an observation from before
     * it is suppressed, since that speech is what the parent just dismissed. A restart can only
     * extend suppression, never shorten a cooldown already running from a later point.
     */
    fun restartCooldown(atMs: Long) {
        cooldownStartMs = maxOf(cooldownStartMs ?: atMs, atMs)
        window.clear()
    }

    /** Clears accumulated evidence and any active cooldown. */
    fun reset() {
        window.clear()
        cooldownStartMs = null
        lastTimestampMs = null
    }
}
