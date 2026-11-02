#!/usr/bin/env python3
"""DSO v4.3 — Topological Score Grammar: implementation companion (light theme).

Expanded edition: every weight and rule documented with its mathematical role,
the effect of raising/lowering it on the methodology, and the audible result.
"""
import json, sys, os
sys.path.insert(0, "/home/claude/gov"); sys.path.insert(0, "/mnt/project")
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, HRFlowable,
                                Table, TableStyle, NextPageTemplate, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("DejaVuMono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))

import pattern_language as PL
from schema_lexicon import LEXICON, SchemaMarkov
from preference_rules import RuleRegistry
import score_governor as SG

WHITE = HexColor("#ffffff"); PANEL = HexColor("#f4f5f7"); PANEL2 = HexColor("#e6e9ee")
H1C = HexColor("#1f3a5f"); H2C = HexColor("#0f766e"); BODY = HexColor("#1c1c1e"); DIM = HexColor("#5b5f66")
GRID = HexColor("#c7ccd3"); WARN = HexColor("#8a3b12")
PAGE_W, PAGE_H = letter; L_W, L_H = landscape(letter); M = 0.8 * inch

def bg_factory(w, h):
    def bg(c, doc):
        c.saveState(); c.setFillColor(WHITE); c.rect(0, 0, w, h, fill=1, stroke=0)
        c.setFont("Helvetica", 8); c.setFillColor(DIM)
        c.drawString(M, 0.45 * inch, "Audible Lava / DSO v4.3 — Topological Score Grammar: implementation companion")
        c.drawRightString(w - M, 0.45 * inch, f"{doc.page}"); c.restoreState()
    return bg

S = {
 "title": ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=21, leading=26, textColor=H1C, spaceAfter=4),
 "subtitle": ParagraphStyle("st", fontName="Helvetica", fontSize=10.5, leading=14, textColor=H2C, spaceAfter=14),
 "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14.5, leading=18, textColor=H1C, spaceBefore=14, spaceAfter=5),
 "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=H2C, spaceBefore=10, spaceAfter=4),
 "body": ParagraphStyle("b", fontName="Helvetica", fontSize=9.3, leading=13.2, textColor=BODY, spaceAfter=7),
 "bullet": ParagraphStyle("bl", fontName="Helvetica", fontSize=9.3, leading=13.2, textColor=BODY, leftIndent=15, bulletIndent=4, spaceAfter=4),
 "cell": ParagraphStyle("c", fontName="Helvetica", fontSize=7.2, leading=9.2, textColor=BODY),
 "cellsm": ParagraphStyle("csm", fontName="Helvetica", fontSize=6.8, leading=8.6, textColor=BODY),
 "cellhead": ParagraphStyle("ch", fontName="Helvetica-Bold", fontSize=7.8, leading=9.8, textColor=H1C),
 "cellsec": ParagraphStyle("cs", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=H2C),
}
def P(t, s="body"): return Paragraph(t, S[s])
def B(t): return Paragraph(f'<bullet color="#0f766e">&bull;</bullet>{t}', S["bullet"])
def rule_(): return HRFlowable(width="100%", thickness=0.75, color=H2C, spaceBefore=2, spaceAfter=8)
def c(t): return f'<font face="DejaVuMono" color="#8a3b12" size="8.4">{t}</font>'
def cc(t): return f'<font face="DejaVuMono" color="#8a3b12" size="6.6">{t}</font>'
def strong(t): return f'<b><font color="#1f3a5f">{t}</font></b>'
def cell(t, s="cell"): return Paragraph(t, S[s])
def csm(t): return Paragraph(t, S["cellsm"])

def table(header, rows, widths, sec_rows=()):
    data = [[cell(h, "cellhead") for h in header]] + rows
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), PANEL2), ("GRID", (0, 0), (-1, -1), 0.4, GRID),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PANEL])]
    for r in sec_rows:
        st += [("BACKGROUND", (0, r), (-1, r), PANEL2), ("SPAN", (0, r), (-1, r))]
    t.setStyle(TableStyle(st)); return t

def sec(title, n):
    return [cell(title, "cellsec")] + [cell("")] * (n - 1)

story = []
story.append(P("Topological Score Grammar — Implementation Companion", "title"))
story.append(P("DSO v4.3: the complete weight and rule reference — what each parameter does mathematically, "
               "how changing it alters the methodology, and what it does to the sound", "subtitle"))
story.append(rule_())

# ═════════════ 1. Signal path ═════════════
story.append(P("1 &nbsp;How a weight becomes a sound", "h1")); story.append(rule_())
story.append(P("Every weight in this system sits at one of five points on a single path. Knowing which point a "
               "weight occupies tells you what kind of change to expect from turning it: rule weights change "
               "<i>what the system believes</i>, structural weights change <i>how readily it changes its mind</i>, "
               "and only the last stage changes anything directly audible."))
story.append(B(f"{strong('1. Evidence.')} Tracker state and {c('SystemSummary')} produce an evidence dict each "
               "frame: z-scores, BOCPD posteriors, saddle strengths, periodicity strengths, entropies. No weights here."))
story.append(B(f"{strong('2. Rules.')} Each rule f<sub>r</sub>(evidence, candidate) &rarr; [0,1] scores how well a "
               f"candidate explains the evidence. {strong('Rule weights')} w<sub>r</sub> set how much each rule "
               "contributes: s<sub>i</sub> = &Sigma;<sub>r</sub> w<sub>r</sub>&middot;f<sub>r</sub>. Relative size is "
               "all that matters — doubling every weight in a module changes nothing."))
story.append(B(f"{strong('3. Commitment.')} Viterbi accumulates V<sub>i</sub> = [log s<sub>i</sub> + "
               f"max<sub>j</sub>(V<sub>j</sub> &minus; cost)]&middot;forget. {strong('switch_cost')} and "
               f"{strong('forget')} decide how much history resists a change of interpretation; "
               f"{strong('temperature')} converts V into the posterior softmax(V/T)."))
story.append(B(f"{strong('4. ScoreState.')} Committed interpretations plus continuous scalars (tension, ambiguity, "
               f"prominence, pitch field, clock parameters) are written into {c('ScoreState')}."))
story.append(B(f"{strong('5. Synthesis.')} {c('_compute_targets')} adds one summed layer (gravitation, prominence "
               "mixer, roughness, boundary attack), the pattern scheduler emits OSC bundles against the elastic "
               "clock, and SuperCollider smooths every parameter. This is the only stage where a change is "
               "immediately audible."))
story.append(P(f"{strong('Practical consequence.')} If the score sounds wrong, first ask which stage is at fault. "
               "Structure appearing in the wrong places is stage 2 (rule weights). Structure that thrashes or never "
               "moves is stage 3 (switch_cost / forget / temperature). Structure that is correct but inaudible is "
               "stage 5 (score_gain and the mapping depths). Turning rule weights to fix a stage-5 problem is the "
               "most common way to get lost."))

# ═════════════ 2. Structural weights ═════════════
story.append(NextPageTemplate("wide")); story.append(PageBreak())
story.append(P("Table A — Structural weights: mechanism, effect of change, audible result", "h1")); story.append(rule_())
story.append(P("These weights do not describe the lamp. They describe how much evidence the system requires before "
               "it changes its mind, and how sharply it commits once it has. They are the weights most likely to be "
               "mis-set, because their effect is indirect.", "body"))

rows = []
def W(name, default, mech, up, down, sound, rng, note=""):
    rows.append([cell(cc(name)), cell(str(default)), csm(mech), csm(up), csm(down), csm(sound), cell(rng),
                 csm(note)])

rows.append(sec("ROLE WEIGHTING (softmax over contributors; replaces election entirely)", 8))
W("roles.temperature", 0.6,
  "T in w<sub>i</sub> = exp(s<sub>i</sub>/T)/&Sigma;exp(s<sub>j</sub>/T) over all blobs AND virtual signals, per role. "
  "Role scores live in [0,1], so T bounds the maximum weight ratio: at T=1.8 the widest possible ratio is "
  "e<sup>0.56</sup>&asymp;1.75, i.e. effectively uniform.",
  "Weights flatten toward uniform. Every blob governs equally; the role signal becomes an average of the "
  "population rather than a selection.",
  "Weights concentrate on the highest-scoring contributor. Approaches hard election as T&rarr;0, reintroducing "
  "the churn the softmax exists to avoid.",
  "High T: a smooth, undifferentiated texture — no blob audibly 'leads'. Low T: individual blobs clearly take "
  "structural roles, but role handovers become abrupt and the crossfade property is lost.",
  "0.3–1.0",
  "In the logged runs this was raised to 1.8, which is past the point of doing anything. If you want roles to be "
  "MORE distinct, go DOWN."),
W("roles.virtual_signal_bias", 0.5,
  "Multiplier on the role scores of system-level contributors (sys:ke, sys:flux, sys:occupancy_entropy, "
  "sys:kuramoto, sys:tree, sys:equilibrium) before the softmax.",
  "Aggregate signals outcompete individual blobs for roles. Governance becomes whole-lamp rather than per-blob.",
  "Blobs dominate; the aggregate signals contribute only when no blob resolves a periodicity.",
  "High: meter follows the lamp's global convection overturn — steadier, more 'breathing'. Low: meter follows "
  "individual blob cycles — more irregular, more responsive to single events.",
  "0.2–1.5",
  "Raise this when blob lifetimes are short: aggregate signals are the only stable oscillators in that regime."),

rows.append(sec("COMMITMENT (per Viterbi module: grouping, meter, harmony)", 8))
W("&lt;module&gt;.switch_cost", "1.5 / 3.0 / 2.0",
  "Penalty subtracted from V when the committed candidate changes. This IS the hysteresis — it emerges from the "
  "optimisation rather than being a bolted-on threshold.",
  "Interpretations persist longer. The module needs sustained contrary evidence to switch.",
  "The module tracks evidence closely and may thrash frame to frame.",
  "High: long stable phrases, steady tempo, slow harmonic rhythm. Low: restless — boundaries everywhere, tempo "
  "jumping, harmony stepping constantly.",
  "0.5–5.0",
  "CAVEAT: switch_cost only protects a candidate that stays IN the beam. If the proposal step evicts the "
  "incumbent, lowering (or raising) this does nothing — the observed meter behaviour."),
W("&lt;module&gt;.forget", "0.70 / 0.98 / 0.95",
  "Per-frame decay applied to V. Sets the effective memory: V converges to a fixed point of roughly "
  "log(s)&middot;f/(1&minus;f), so f=0.98 gives ~50 frames of memory, f=0.7 gives ~3.",
  "Longer memory. Past evidence keeps weighing on the present; the module is slow and stable.",
  "Short memory. At f&asymp;0.2 the module degenerates to a per-frame argmax with no hysteresis at all, "
  "regardless of switch_cost.",
  "High: structure that feels considered and slow to respond. Low: structure that tracks the lamp instant by "
  "instant, which reads as arbitrary rather than grammatical.",
  "0.90–0.99",
  "Grouping is deliberately low (0.7) because a boundary is an instantaneous judgement; meter and harmony are "
  "high because they are ongoing states."),
W("&lt;module&gt;.temperature", "0.5 / 1.0 / 0.8",
  "T in the posterior softmax(V/T). Does not affect which candidate is committed — only how sharply probability "
  "mass concentrates on it.",
  "Posterior flattens. More mass on rivals, so any consumer that blends by posterior crossfades more.",
  "Posterior sharpens toward one-hot. Blended consumers snap.",
  "This is the crossfade control. High T in harmony makes the predicted next triad audibly pre-sound (the "
  "Imagination phase); low T makes chord changes arrive without warning.",
  "0.3–1.5",
  "The only weight whose entire purpose is the emergent-crossfade property. Raise it if transitions feel abrupt."),
W("grouping.commit_threshold", 0.55,
  "A boundary is emitted when the boundary posterior exceeds this AND a 1.5 s refractory has elapsed.",
  "Fewer, stronger boundaries. Sections get longer.",
  "More boundaries, including weak ones. Phrases shorten.",
  "Directly sets phrase length, and therefore how often the global gestures (envelope reset, register shift, "
  "texture thinning, pattern fill) fire.",
  "0.35–0.75",
  "CAVEAT: with a two-candidate beam, argmax==boundary only when posterior&gt;0.5, so any value below 0.5 is "
  "unreachable unless the argmax requirement is dropped. This is why lowering it to 0.19 produced no boundaries."),

rows.append(sec("CONTINUOUS ESTIMATION", 8))
W("meter.pf_mix", 0.5,
  "Blend of the discrete beam and the particle filter: period = (1&minus;mix)&middot;beam + mix&middot;particle.",
  "Tempo drifts continuously; the particle cloud tracks smoothly between candidate periods.",
  "Tempo steps between discrete candidate values; no intermediate tempi exist.",
  "0: the pulse changes in audible jumps. 1: the pulse breathes, never settling exactly. Around 0.5 reads as "
  "'alive but not vague'.",
  "0.3–0.7",
  "If particle spread stays wide (see governor_runtime.json), the PF has not converged and mixing it in only "
  "adds noise — lower this until the evidence improves."),
W("meter_pf.n_particles", 150,
  "Size of the particle cloud over (period, phase). Resampling triggers when effective sample size falls below "
  "N/2.",
  "Better coverage of the continuous space; smoother estimate; more CPU.",
  "Cloud may collapse onto a local optimum and lose track of the true period.",
  "Rarely audible on its own. Too few particles shows up as tempo suddenly locking to a wrong value.",
  "80–400",
  "Cost is linear; 150 costs well under a millisecond per frame."),
W("timespan.smoothing", 0.9,
  "EMA pole on per-blob prominence: p &larr; a&middot;p + (1&minus;a)&middot;raw. Time constant &asymp; "
  "1/(1&minus;a) frames (~0.7 s at a=0.9, 15 fps).",
  "Prominence changes slowly; the foreground/background mix is stable.",
  "Prominence reacts immediately to per-frame fluctuation.",
  "High: voices hold their place in the mix; the texture has a settled hierarchy. Low: the mix churns, which "
  "reads as nervous rather than structured.",
  "0.85–0.97", ""),
W("prolongation.smoothing", 0.92,
  "EMA pole on the tension composite. ~4 s time constant at 15 fps.",
  "Tension becomes a long-form arc.",
  "Tension tracks instantaneous surprisal and reads as twitchy.",
  "This sets the timescale of the piece's dramatic shape — how long a build takes and how long a release lasts.",
  "0.88–0.98",
  "Raise toward 0.97 for installation-scale arcs; lower toward 0.85 for a reactive, event-driven feel."),
W("harmony.step_rate_gain", 4.0,
  "Multiplier converting diagram-Wasserstein per second into step budget; the walker moves when the budget "
  "crosses 1.0.",
  "Harmony steps more often for the same topological change rate.",
  "Harmony becomes static, moving only during large reconfigurations.",
  "Sets harmonic rhythm. High: chord changes every few seconds. Low: one field held across a whole section.",
  "1.0–12.0",
  "The most direct control over how 'harmonic' the piece sounds versus how 'drone-like'."),
W("bocpd.hazard", 0.0083,
  "Prior probability per sample that a changepoint occurs, i.e. 1/expected run length. 0.0083 &asymp; one "
  "changepoint per 120 samples.",
  "Changepoints are expected more often; the detector fires more readily and on smaller shifts.",
  "The detector demands a large, sustained shift before declaring a change.",
  "Feeds the strongest grouping rule, so this indirectly sets how event-driven the phrasing is.",
  "1/60 – 1/300",
  "Set it to roughly 1/(expected frames between real regime changes) rather than tuning by ear."),

secs = [i + 1 for i, r in enumerate(rows) if r[0].style.name == "cs"]
story.append(table(["Weight", "Default", "Mechanism", "Raising it", "Lowering it", "Audible effect", "Range", "Note"],
                   rows, [1.25*inch, 0.62*inch, 1.85*inch, 1.5*inch, 1.5*inch, 1.85*inch, 0.55*inch, 1.6*inch], secs))

# ═════════════ 3. Rule weights ═════════════
story.append(PageBreak())
story.append(P("Table B — Rule weights: evidence, scoring, effect of change", "h1")); story.append(rule_())
story.append(P("Each rule maps evidence to [0,1] for a candidate. Only RELATIVE weights matter within a module. "
               "The 'shifts the score toward' column is the practical question: if you raise this weight, what kind "
               "of structure does the system start finding?", "body"))

rows = []
def R(module, name, weight, evidence, scoring, up, sound):
    rows.append([cell(cc(name)), cell(str(weight)), csm(evidence), csm(scoring), csm(up), csm(sound)])

rows.append(sec("GROUPING — evidence for the boundary event. Score s is z-scored, so only RELATIVE weights matter; the hazard supplies phrase length separately (see section 2)", 6))
R("grouping", "gap", 1.5, "Time since the previous blob/system event, vs. the running mean phrase length.",
  "Rises with elapsed time &times; cue count. GPR2 analog (proximity).",
  "Boundaries drift toward moments after a lull — silence-then-event phrasing.",
  "Phrases articulated by rests. The most 'classical' of the boundary cues.")
R("grouping", "change", 2.0, "Max |z| across KE, occupancy entropy, flux, coherence.",
  "(z&minus;1)/3 clipped to [0,1]. GPR3 analog (change).",
  "Boundaries land on abrupt shifts in lamp activity, whatever their cause.",
  "Phrasing that follows the lamp's energy contour: a plume onset becomes a new section.")
R("grouping", "saddle", 2.0, "Persistence of the branch absorbed at a new merge-tree saddle.",
  "min(1, persistence/60). The only topologically-grounded boundary cue.",
  "Boundaries follow genuine hierarchical merges rather than surface activity.",
  "The most structurally defensible phrasing — deep saddles become section breaks, shallow ones motive "
  "articulations. Raise this for the paper's central claim to be audible.")
R("grouping", "changepoint", 2.5, "Max BOCPD posterior across all monitored signals.",
  "min(1, 2&middot;p). One mechanism, many signals.",
  "Boundaries become statistically principled: they land where a regime genuinely changed.",
  "Currently the highest-weighted cue, so phrasing is primarily regime-driven. Lower it if phrasing feels "
  "disconnected from anything you can see in the lamp.")
R("grouping", "regime", 1.0, "z of conditional entropy and of diagram-Wasserstein.",
  "(z&minus;1)/3 clipped. Entropic and topological regime shifts.",
  "Boundaries at moments when the system's predictability changes, not its activity.",
  "Subtle: phrasing that responds to the lamp becoming more or less predictable rather than more or less busy.")
R("grouping", "relational", 1.0, "Pairwise-distance variance z, collinearity events, COM midline crossings.",
  "Weighted sum, clipped.",
  "Boundaries follow configuration changes — blobs rearranging rather than individual blobs acting.",
  "Phrasing driven by the constellation as a whole. Good when blob count is stable but positions shift.")
R("grouping", "coincidence", 1.5, "Number of independent cue types firing within the window.",
  "(n&minus;1)/3 clipped. GTTM's own insight: aligned cues make strong boundaries.",
  "Only multi-cue moments become boundaries; single-cue moments are suppressed.",
  "Fewer but much more convincing boundaries. Raise this if boundaries feel arbitrary — it is the cheapest way "
  "to make phrasing feel deliberate.")
R("grouping", "regularity", "—", "Ratio of elapsed time to mean phrase length.",
  "SUPERSEDED in v4.4; still registered for reference but dropped by HazardGroupingModule.IGNORED_RULES.",
  "Nothing. The learned Gamma hazard represents phrase length directly, and its shape parameter expresses "
  "regularity better than a fixed weight could.",
  "To make phrasing more periodic no weight is needed: the shape parameter rises on its own when the lamp "
  "produces regular intervals. Watch grouping.phrase_model.shape in the runtime JSON.")

rows.append(sec("METER — tactus period and beat grouping (beam: (period, beats) hypotheses)", 6))
R("meter", "periodicity", 2.0, "Autocorrelation peak strength of every elected signal, at the candidate's period "
  "and its octave/triple multiples.",
  "Best matching strength, weighted by the signal's role weight.",
  "The clock follows whichever observable is genuinely most periodic.",
  "Tempo that tracks a real oscillation in the lamp. Dominant weight by design.")
R("meter", "london_prior", 1.0, "None — a prior on the candidate period itself.",
  "Gaussian in log period centred at 0.7 s.",
  "Tempi pull toward a comfortable human tactus regardless of evidence.",
  "Keeps the pulse tappable. Lower it to let the lamp's own slow cycles set unusual tempi.")
R("meter", "cardinality", 1.0, "Blob count.",
  "1.0 when beats == clip(n, 2, 6), else 0.5.",
  "Meter follows population: three blobs favour triple grouping.",
  "Directly audible as duple vs triple feel. A conceptually neat mapping — the lamp's population becomes the "
  "time signature.")
R("meter", "ratio_evidence", 1.0, "Ratio of the two slowest resolved blob periods.",
  "Near 3:2 favours 3- or 6-beat bars; otherwise 2 or 4.",
  "Polyrhythmic relationships in the lamp become meter, not just tempo.",
  "Hemiola baked into the bar when two blobs cycle at 3:2. The most musically interesting rule in the set; "
  "requires long blob lifetimes to fire at all.")

rows.append(sec("METER PARTICLE FILTER — continuous (period, phase)", 6))
R("meter_pf", "onset_alignment", 2.0, "Blob turnaround times over the last 8 s.",
  "Mean closeness of onsets to the particle's beat grid.",
  "Phase locks to actual physical events rather than to an abstract period.",
  "Beats coincide with visible blob turnarounds — the strongest sense that the music is 'watching' the lamp.")
R("meter_pf", "pf_continuity", 1.0, "The discrete beam's committed period.",
  "Gaussian in log ratio.",
  "The particle estimate stays tethered to the discrete hypothesis.",
  "Prevents the continuous and discrete layers from disagreeing audibly. Lower it to let the PF explore.")
R("meter_pf", "pf_london", 0.5, "None — prior on particle period.",
  "Same Gaussian as london_prior.",
  "Particles avoid drifting to implausible tempi.",
  "A safety term; rarely worth changing.")

rows.append(sec("HARMONY — the Tonnetz walk (beam: current triad + P/L/R neighbours)", 6))
R("harmony", "parsimony", 2.0, "Distance in P/L/R steps from the current triad.",
  "1/(1+d). Cohn's voice-leading parsimony as a preference.",
  "Only the smoothest moves are taken; the field creeps.",
  "Chord changes that sound like slow morphing rather than progression. Matches wax phenomenology; lower it for "
  "more dramatic harmonic motion.")
R("harmony", "step_gate", 2.0, "Accumulated step budget from diagram-Wasserstein.",
  "Rewards staying put until the budget is spent, then rewards moving.",
  "Harmony moves strictly when the topology moves.",
  "The literal implementation of 'the harmony moves when the lamp does'. Lower it and harmony drifts on its own "
  "schedule, breaking the coupling.")
R("harmony", "direction", 1.5, "Sign of the Wasserstein-to-equilibrium gradient.",
  "Rewards moves away from home when leaving equilibrium, toward home when returning.",
  "Distance-from-tonic tracks distance-from-rest closely.",
  "The Q4 isomorphism made audible: the music is far from home exactly when the lamp is far from rest. This is "
  "the weight that carries the piece's central claim.")
R("harmony", "size_license", 1.0, "max(tension, normalised surprisal).",
  "Compound (2-step) moves scored only under high tension.",
  "Distant harmonic moves require dramatic justification.",
  "Meyer operationalised: a surprising physical transition licenses a surprising chord. Lower it and large leaps "
  "happen arbitrarily.")
R("harmony", "content_fit", 1.0, "Pitch classes currently sounding.",
  "Fraction of sounding pcs contained in the candidate triad.",
  "The field follows what the texture is already playing.",
  "Harmony feels emergent from the voices rather than imposed on them. Temperley's key-finding, inverted.")

rows.append(sec("TIME-SPAN PROMINENCE — head vs ornament (weighted sum, no beam)", 6))
R("timespan", "integrated_persistence", 1.0, "&Sigma; H0 persistence over the blob's whole life.",
  "Normalised by the running max.",
  "Long-standing blobs outrank briefly-bright ones.",
  "Sustained presences become the foreground. Needs long blob lifetimes to mean anything.")
R("timespan", "tree_depth", 1.0, "Leaf depth in the corrected merge tree.",
  "1 &minus; depth/max_depth: root-adjacent = head.",
  "Structural level, not lifetime, decides prominence.",
  "The Schenkerian mapping: background material is slow, stable and central; deep leaves get diminution. "
  "Independent of the lifecycle problem, so raise this when blob lifetimes are short.")
R("timespan", "coupling", 0.8, "|corr(blob vy, aggregate flux)|.",
  "Hub blobs that drive the system score higher.",
  "Prominence follows influence rather than size.",
  "Deeply ecosystemic and reviewer-friendly: the voice that leads is the blob the others follow.")
R("timespan", "recurrence", 0.6, "Re-identified blob returning after absence.",
  "Binary, gated on tracker re-ID.",
  "Returning material is treated as thematic.",
  "A returning theme regains its former prominence. Dies entirely if reident_max_age is tightened.")
R("timespan", "energy", 0.6, "Work integral &Sigma;&frac12;&middot;area&middot;speed&sup2;&middot;dt.",
  "Normalised by running max.",
  "Physically active blobs outrank passive ones.",
  "The mix follows exertion. Complements persistence, which follows brightness.")
R("timespan", "surface", 1.2, "Events on strong beats of the governed clock and on chord tones of the current field.",
  "0.5&middot;(on_beat + in_field). What GTTM's actual time-span rules use.",
  "Prominence becomes a joint function of physics AND the score's own grammar.",
  "Self-referential in the Di Scipio sense: a voice is a head partly because it is playing in time and in key. "
  "The only prominence term immune to blob lifetime — raise it when tracking is unstable.")

rows.append(sec("PROLONGATIONAL TENSION — the composite scalar (weighted sum, no beam)", 6))
R("prolongation", "surprisal", 1.0, "&minus;log&#8322; P(symbol | previous symbol), normalised by 6 bits.",
  "Meyer: expectation violated &rarr; tension.",
  "Tension spikes on unexpected physical transitions.",
  "Sharp, event-driven drama. Raise for a reactive piece.")
R("prolongation", "w_equilibrium", 1.0, "Marginal-EMD between current occupancy and the equilibrium grid.",
  "Distance from physical rest.",
  "Tension tracks how far the wax is from its settled state.",
  "Slow, arch-shaped tension following the convection cycle. The most 'installation-scale' term.")
R("prolongation", "ke", 0.8, "Total kinetic energy, normalised by a running max.",
  "Raw activity level.",
  "Tension follows how busy the lamp is.",
  "The most immediately legible mapping for an audience watching the lamp.")
R("prolongation", "cond_entropy", 0.8, "H(next | current) of the symbolised macrostate.",
  "Uncertainty about the immediate future.",
  "Tension rises when the system becomes unpredictable.",
  "Anxious, unsettled quality independent of activity. CAVEAT: saturates at 1.0 if the Markov alphabet is too "
  "large for the run length, in which case this term is a constant and contributes nothing.")

secs = [i + 1 for i, r in enumerate(rows) if r[0].style.name == "cs"]
story.append(table(["Rule", "Weight", "Evidence it reads", "How it scores", "Raising it shifts the score toward",
                    "Audible / methodological effect"],
                   rows, [1.15*inch, 0.5*inch, 1.9*inch, 1.75*inch, 1.9*inch, 2.55*inch], secs))

# ═════════════ 4. Interactions & failure modes ═════════════
story.append(PageBreak())
story.append(P("Table C — Interactions, failure modes, and what to turn", "h1")); story.append(rule_())
story.append(P("Weights are not independent. This table lists the couplings that matter in practice and the "
               "diagnostic signature of each failure, as observed in governor_runtime.json / governor_log.txt.", "body"))
rows = []
def F(sym, sig, cause, fix):
    rows.append([csm(sym), csm(sig), csm(cause), csm(fix)])

rows.append(sec("COUPLINGS — weights that fight each other", 4))
F("switch_cost vs. beam membership",
  "meter.switches keeps rising however switch_cost is set.",
  "switch_cost only protects a candidate that remains IN the beam. The meter proposal step sorts by raw "
  "periodicity and truncates, evicting the incumbent, so its accumulated V is discarded and re-entry starts at "
  "&minus;switch_cost.",
  "Retain the incumbent after sorting; quantise candidate periods onto a log grid so keys repeat frame to frame. "
  "Only then does switch_cost behave as documented.")
F("[v4.2, RESOLVED] commit_threshold vs. beam size",
  "Lowering commit_threshold below 0.5 produced no additional boundaries.",
  "With two candidates, argmax==boundary is exactly posterior&gt;0.5, and commitment required both — two knobs "
  "that were one knob, with half its range unreachable.",
  "Fixed in v4.4: grouping is an event model with no threshold and no argmax. See section 2.")
F("forget vs. switch_cost",
  "Structure thrashes at any switch_cost.",
  "forget dominates: at f=0.2 the V fixed point is ~0.4 versus ~78 at f=0.98, so accumulated evidence is "
  "negligible next to any switching penalty.",
  "Keep forget &ge; 0.9 for meter and harmony. Tune switch_cost only after forget is sane.")
F("roles.temperature vs. blob count",
  "Role weights near-uniform however T is set.",
  "Role scores are bounded in [0,1]; with few contributors and high T the softmax cannot differentiate.",
  "Lower T to 0.3–0.5 when blob count is small.")
F("timespan lifetime terms vs. tracker settings",
  "Prominence ranking looks arbitrary.",
  "integrated_persistence, energy and recurrence all need long-lived, re-identified blobs. Short lifetimes "
  "starve three of six terms.",
  "Raise timespan.rules.surface (immune to lifetime) and tree_depth, or fix tracking first.")
F("cond_entropy vs. alphabet size",
  "ambiguity pinned at 1.0 for the whole run.",
  "Dirichlet prior mass (0.5 &times; |alphabet|) swamps observed counts, so H stays at log&#8322;|alphabet| and the "
  "fixed /3.0 normaliser saturates.",
  "Lower prior to ~0.05, reduce the alphabet, and normalise by log&#8322; of the observed alphabet.")

rows.append(sec("FAILURE MODES — signature, cause, weight to turn", 4))
F("No boundaries ever fire",
  "grouping.p_boundary near zero every frame; n_boundaries stays 0.",
  "Either the hazard never rises (prior_phrase_secs longer than the run), or evidence never exceeds its own "
  "median because the rules return near-constant values, so z&asymp;0 and Phi(z)&asymp;0.5 forever.",
  "Check grouping.evidence_model.mad — near zero means the rules are not discriminating. Lower "
  "prior_phrase_secs; raise changepoint and saddle relative to the rest.")
F("Boundaries everywhere",
  "n_boundaries climbing fast; phrase_model.mean_secs collapsing toward a few seconds.",
  "Positive feedback: short intervals raise the hazard, producing shorter intervals. Usually driven by "
  "evidence_rate being too LOW, so a sustained busy stretch reads as continuously remarkable.",
  "Raise evidence_rate so the null distribution tracks the new activity level; lower bocpd.hazard. The interval "
  "model is self-correcting once it catches up.")
F("Phrase model will not leave the prior",
  "phrase_model.mean_secs stuck near 25 s with n_observed low.",
  "Too few boundaries have fired for the prior to wash out (it needs 5–6).",
  "Not a fault in itself; if boundaries are genuinely too rare, treat it as the row above.")
F("Tempo never settles",
  "meter.switches &gt;2/s; meter_pf spread.period wide.",
  "Beam eviction plus low forget; or no signal is actually periodic (meter_source reads 'blobs').",
  "Fix beam retention; raise forget to 0.98; raise roles.virtual_signal_bias so aggregate signals can win the "
  "meter role.")
F("Harmony drones on one triad",
  "distance_home constant; harmony.switches flat.",
  "step_rate_gain too low, or diagram-Wasserstein near zero because the merge tree is not changing.",
  "Raise step_rate_gain; verify the tree is being computed on detection frames.")
F("Harmony wanders aimlessly",
  "distance_home random-walking; last_op mostly compounds.",
  "direction weight too low relative to parsimony; size_license too permissive.",
  "Raise direction to 2.5+; raise parsimony; lower size_license.")
F("Tension is a flat line",
  "tension confined to a narrow band for the whole run.",
  "cond_entropy saturated (constant term), and prolongation.smoothing high enough to average the rest.",
  "Fix the Markov normalisation; lower smoothing toward 0.88; raise w_equilibrium for arc shape.")
F("Meter is rigidly quantised",
  "elastic pinned near 1.0 every frame.",
  "Kuramoto order parameter returning ~1.0 — in v4.2 from the atan2 phase proxy, and from defaulting a lone "
  "oscillator to 1.0.",
  "v4.3 fixes both (true per-blob phase; 0.0 when fewer than two oscillators resolve). Watch "
  "rel_kuramoto_n to confirm blobs are actually contributing.")
F("Bass stutter drones continuously",
  "Many slots reading synth_kind 'bass'.",
  "Exit thresholds looser than entry in every dimension, so the hold region was an unbounded superset.",
  "v4.3: narrowed hold region, 6 s hard cap, 3-frame exit debounce, 2-voice cap, 3 s retrigger cooldown.")

secs = [i + 1 for i, r in enumerate(rows) if r[0].style.name == "cs"]
story.append(table(["Coupling / symptom", "Diagnostic signature", "Mechanism", "What to change"],
                   rows, [1.7*inch, 2.3*inch, 3.1*inch, 2.6*inch], secs))

# ═════════════ 5. Tuning procedure ═════════════
story.append(NextPageTemplate("portrait")); story.append(PageBreak())
story.append(P("2 &nbsp;Grouping is an event model (v4.4)", "h1")); story.append(rule_())
story.append(P("A boundary is an <i>event</i>; &ldquo;continue&rdquo; is a <i>state</i>. Modelling them as two "
               "competing states in a Viterbi beam was a category error, and every v4.2 symptom followed from it. "
               f"The commit test required {c('argmax == boundary')} AND {c('posterior &gt; commit_threshold')}; with "
               "a two-candidate beam argmax is exactly posterior &gt; 0.5, so the threshold did nothing below 0.5 "
               "and the argmax did nothing above it. Two knobs that were one knob, with the useful half of the "
               f"range unreachable &mdash; which is why walking {c('commit_threshold')} from 0.25 to 0.19 produced "
               "no boundaries at all while posteriors sat at 0.46&ndash;0.48. Separately, "
               f"{c('grouping.forget')} had to be detuned to 0.7 (against 0.98 elsewhere) to stop a boundary "
               f"&ldquo;state&rdquo; persisting, a refractory constant had to be bolted on, and a {c('regularity')} "
               "rule had to hand-encode a phrase-length preference the model could not represent."))
story.append(P("v4.4 replaces the beam with a product of two learned distributions:"))
story.append(P("<b>P(boundary at t) &nbsp;=&nbsp; h(&tau;) &middot; &Phi;(z)</b>", "h2"))
story.append(B(f"{strong('h(&tau;)')} &mdash; the hazard of an online Gamma(k, &theta;) fitted to observed "
               "inter-boundary intervals, evaluated at &tau; = time since the last boundary. The piece learns its "
               "own phrase-length distribution. This one term subsumes three pieces of v4.2 configuration: the "
               "refractory period (the hazard is near zero just after a boundary by construction), the "
               f"{c('regularity')} rule (the hazard peaks near the typical phrase length), and the threshold's role "
               "of setting boundary rate."))
story.append(B(f"{strong('&Phi;(z)')} &mdash; the normal CDF of z = (s &minus; &mu;)/&sigma;, where s is the "
               "weighted rule sum and &mu;, &sigma; come from an online median and MAD. Evidence becomes "
               "scale-free: the absolute magnitude of the rule weights cancels, and so does lamp brightness. Only "
               "<i>relative</i> rule weights matter now, and a re-calibrated detector needs no re-tuning here."))
story.append(P("The product is a soft AND, closer to how GTTM's preference rules actually interact than a "
               "sum-and-threshold: a boundary needs both <i>we are due</i> and <i>something happened</i>. Strong "
               "evidence early in a phrase will not fire, and being badly overdue with nothing happening will not "
               "either."))
story.append(P(f"{strong('Commitment is stochastic')} &mdash; a boundary is drawn with probability P per frame "
               "rather than tested against a threshold. That matches the project's principle that structure should "
               "emerge from the probabilistic model rather than from explicit machinery, and it yields graded "
               "boundaries for free: strength and depth follow from how strongly the two factors agreed, so a "
               "high-z, well-overdue moment reads as a section break and a marginal one as motive-level "
               "articulation."))
story.append(P(f"{strong('Two robustness details that testing forced.')} First, the prior is blended by a weight "
               "that decays with observation count rather than folded into the running sums: folding it in lets its "
               "variance (mean&sup2;/shape, which is large) contaminate the shape estimate long after its mean has "
               "washed out &mdash; the fitted shape read about 3 on 60 observations that truly implied about 60. "
               "Second, the MAD update <i>trims</i> rather than winsorizes: samples beyond 4&sigma; do not update "
               "the scale at all. Winsorizing was not enough, because a run of consecutive extremes still walked "
               "the scale upward, and the spikes the model exists to find would then desensitise it to the next "
               "one. The median still tracks every sample, since its sign step is robust by construction."))
story.append(P(f"{strong('Configuration deleted:')} {c('commit_threshold')}, {c('switch_cost')}, {c('forget')}, "
               f"{c('temperature')}, the 1.5&nbsp;s refractory constant, and the {c('regularity')} rule weight. "
               f"{strong('Added:')} {c('prior_phrase_secs')} (25&nbsp;s) and {c('evidence_rate')} (0.01). Four "
               "learned quantities replace them: Gamma shape and scale, evidence median and MAD."))
story.append(P(f"{strong('What to watch.')} {c('governor_runtime.json')} now reports "
               f"{c('grouping.phrase_model')} (mean_secs, shape, scale, n_observed) and "
               f"{c('grouping.evidence_model')} (median, mad, n). A shape near 1 means memoryless, Poisson-like "
               f"phrasing; a large shape means near-periodic phrasing. {c('boundary_posterior')} now reports "
               "&Phi;(z), the evidence probability, because the per-frame P is tiny by construction (a phrase is "
               "hundreds of frames) and would be useless on a HUD."))

story.append(PageBreak())
story.append(P("3 &nbsp;Tuning procedure", "h1")); story.append(rule_())
story.append(P("Tune in this order. Each stage depends on the one before it being correct, and tuning out of order "
               "is how weights end up compensating for each other's errors."))
story.append(B(f"{strong('0. Verify the substrate.')} Blob lifetimes in the tens of seconds, "
               f"{c('rel_kuramoto_n')} &ge; 2, {c('meter_source')} naming a real signal, ambiguity not pinned at "
               "1.0. If any of these fail, no weight setting will help — the evidence itself is wrong."))
story.append(B(f"{strong('1. Phrasing.')} Nothing to set for rate &mdash; the hazard learns it. Shape the "
               f"phrasing with the evidence weights instead: raise {c('saddle')} to follow genuine hierarchical "
               f"merges, {c('changepoint')} to follow statistical regime shifts, {c('coincidence')} to require "
               f"agreement between cues. Adjust {c('prior_phrase_secs')} only if the first minute feels wrong."))
story.append(B(f"{strong('2. Roles.')} Lower {c('roles.temperature')} until you can hear individual blobs taking "
               "distinct structural roles; raise " + c('roles.virtual_signal_bias') + " if no blob is stable enough."))
story.append(B(f"{strong('3. Meter.')} Confirm the elected signal is sensible, then set {c('meter.pf_mix')} for the "
               "step-vs-drift character you want. Leave switch_cost alone until beam retention is fixed."))
story.append(B(f"{strong('4. Harmony.')} {c('step_rate_gain')} for harmonic rhythm, then {c('direction')} vs "
               f"{c('parsimony')} for how strongly the music tracks distance-from-rest."))
story.append(B(f"{strong('5. Tension.')} {c('prolongation.smoothing')} for the timescale of the arc, then the four "
               "term weights for what the arc is made of."))
story.append(B(f"{strong('6. Save.')} Copy the {c('weights')} block of {c('governor_runtime.json')} back into "
               f"{c('governor_weights.json')} — the structures match exactly."))
story.append(P(f"{strong('Live editing.')} In EDIT mode press {c('g')} for the governor family; arrows step, "
               f"{c('=')}/{c('-')} step coarse (&times;10), {c('r')} resets to the hand-set default. Every change is "
               f"mirrored to {c('governor_runtime.json')} immediately with a {c('last_edit')} stamp, so a listening "
               "session is self-documenting."))

# ═════════════ Appendix tables A–G from the v4.2 edition ═════════════
story.append(NextPageTemplate("wide")); story.append(PageBreak())
story.append(P("Appendix D — Pattern protocol, primitives, transformations, query patterns", "h1")); story.append(rule_())
rows = []
rows.append(sec("PROTOCOL / TYPES (pattern_language.py)", 4))
for n, sig, d in [
    ("Span", "Span(start, end)", "Time span in cycles (one cycle = one bar). cycles(), intersect()."),
    ("Event", "Event(start, end, value, velocity, degree, tags)", "A HIT: discrete onset with duration, accent [0,1], optional scale degree, free tags."),
    ("Pattern", "query(span) -> [Event]", "The protocol. Any object with query() is a pattern (Tidal's model)."),
    ("FnPattern", "FnPattern(fn, label); p >> tf", "Pattern from a query function; fluent transform application."),
]:
    rows.append([cell(n), cell(cc(sig)), csm(d), cell("core")])
rows.append(sec("PRIMITIVES", 4))
for n, sig, d in [
    ("silence", "silence()", "No events."),
    ("pure", "pure(value, velocity)", "One whole-cycle event per cycle."),
    ("steps", "steps([1,0,1,1])", "Subdivide each cycle into equal slots; falsy = rest."),
    ("euclid", "euclid(k, n, rotation)", "E(k,n): k hits as evenly as possible over n slots (Bjorklund/Toussaint); E(3,8) = tresillo."),
    ("from_events", "from_events(events, period)", "Explicit events repeating every period (schemas realise into this)."),
    ("sequence", "sequence(p1, p2, ...)", "One pattern per cycle in turn (Tidal cat)."),
    ("stack", "stack(p1, p2, ...)", "Superimpose."),
]:
    rows.append([cell(n), cell(cc(sig)), csm(d), cell("primitive")])
rows.append(sec("TRANSFORMATIONS (registry; Pattern -> Pattern; composable via compose() / apply_spec())", 4))
for name in PL.TRANSFORMS:
    rows.append([cell(name), cell(cc(f"{name}(...)")), csm(PL.TRANSFORM_DOCS.get(name, "")), cell("registered")])
rows.append(sec("QUERY PATTERNS", 4))
for n, sig, d in [
    ("look-ahead", "pattern.query(Span(c_now, c_now+lookahead))", "PatternScheduler renders 250 ms ahead each frame."),
    ("cycle-wise", "pattern.query(Span(c, c+1))", "Used by every/iter/chunk/rev/density."),
    ("spec-driven", 'apply_spec(p, [("degrade", 0.3), ("ply", 2)])', "Serializable syntax governors emit."),
    ("blob pattern", "ScoreGovernor.pattern_for_blob(blob)", "area->k, persistence->n, beat phase->rotation; spec from tension/prominence/boundary."),
    ("schema pattern", "ScoreGovernor.schema_patterns()", "(melody, bass) of the current schema; ornament level = tension."),
]:
    rows.append([cell(n), cell(cc(sig)), csm(d), cell("query")])
secs = [i + 1 for i, r in enumerate(rows) if r[0].style.name == "cs"]
story.append(table(["Name", "Signature", "Semantics", "Kind"], rows, [1.1*inch, 2.5*inch, 5.2*inch, 0.9*inch], secs))

story.append(PageBreak())
story.append(P("Appendix E — Schema lexicon (12 frames)", "h1")); story.append(rule_())
rows = []
for s_ in LEXICON.values():
    rows.append([cell(f"<b>{s_.name}</b>"), cell(s_.origin), csm(str(list(s_.melody))), csm(str(list(s_.bass))),
                 csm(f"{s_.footprint} beats; T={list(s_.tension)}"), csm(s_.doc + " Tags: " + ", ".join(s_.tags))])
story.append(table(["Frame", "Origin", "Melody degrees", "Bass degrees", "Footprint & tension", "Description"],
                   rows, [0.85*inch, 0.85*inch, 1.5*inch, 1.5*inch, 2.0*inch, 3.0*inch]))
m = SchemaMarkov()
story.append(Spacer(1, 6))
story.append(P("Succession bias (governors multiply tagged columns and renormalise): immediate repeats &times;0.35; "
               "resolving&rarr;tension &times;1.6; tension&rarr;resolving &times;1.6; closing&rarr;opening &times;2.0; "
               "rising&rarr;descending &times;1.3. Row for <b>cadence</b>: "
               + ", ".join(f"{k} {v:.2f}" for k, v in sorted(m.distribution('cadence').items(), key=lambda kv: -kv[1])[:6])
               + " …", "body"))

story.append(PageBreak())
story.append(P("Appendix F — Files and integration points (v4.3)", "h1")); story.append(rule_())
rows = []
for f, k, d in [
    ("hazard_grouping.py", "v4.4 NEW", "GammaIntervalModel (learned phrase-length hazard), RobustStandardiser "
     "(trimmed online median/MAD), HazardGroupingModule. Replaces the Viterbi grouping beam."),
    ("system_state.py", "v4.3", "SystemSummary, entropies, EMD/Wasserstein, BOCPD, MarkovExpectation, SystemStateTracker. "
     "v4.3: Kuramoto now uses true per-blob oscillation phase (relational is an instance method); "
     "blob_phase_and_period detrends drift, requires a resolved cycle, and derives dy/dt numerically so the phase "
     "is frame-rate invariant."),
    ("preference_rules.py", "v4.2", "Rule/RuleRegistry, PreferenceRuleModule (Viterbi), ParticleFilterModule, WeightStore, softmax_weights."),
    ("pattern_language.py", "v4.2", "Span/Event/Pattern, primitives, 21 registered transforms, apply_spec, PatternScheduler."),
    ("schema_lexicon.py", "v4.2", "12 schemas, SchemaMarkov, realize(), degree_to_freq."),
    ("harmony_tonnetz.py", "v4.2", "Triad/P/L/R, shortest paths, TonnetzWalker, roughness proxy."),
    ("elastic_clock.py", "v4.2", "ElasticClock (projective, Kuramoto-governed pull, London bounds)."),
    ("score_governor.py", "v4.4", "ScoreState, rules, modules, roles, boundary evidence, merge-tree consumption, "
     "JSON I/O, EDIT hook. v4.4: grouping consumes HazardGroupingModule; commit_threshold / switch_cost / forget / "
     "temperature removed from the grouping weights."),
    ("background_reference.py", "v4.3", "NEW warm-lamp capture: long-window temporally-subsampled PERCENTILE (25th, not median) "
     "recovers background while wax is present in every frame; wall-clock pacing; refine() pass excluding foreground; "
     "occupancy() diagnostic; reset(mode=...)."),
    ("voice_manager.py", "v4.3", "Governor consumption layer (v4.2) plus bass-stutter fixes: narrowed hold region, "
     "6 s hard cap, 3-frame exit debounce, 2-voice cap, 3 s retrigger cooldown, bass state cleared on detach."),
    ("controller.py", "v4.4", "'gov' EDIT family (16 governor params) + step sizes; 'g' key. v4.4: the grouping "
     "params are prior_phrase_secs (1 s steps) and evidence_rate."),
    ("routing.py", "v4.2", "8 governor Sources read from stamped blob attributes."),
    ("merge_tree.py / h0_persistence.py", "v4.2", "Corrected GUDHI stride, exact coface localization, essential class, true saddles."),
    ("test_score_grammar.py", "v4.2", "21 behavioural tests."),
    ("test_v43_fixes.py", "v4.3", "23 tests: bass lifecycle (8), warm background capture (7), Kuramoto phase (8)."),
    ("test_hazard_grouping.py", "v4.4 NEW", "22 tests: Gamma interval model (6), robust standardiser (3), "
     "event-model behaviour (13) — emergent refractory, learned phrase length, scale invariance, sane boundary "
     "rate with no threshold."),
]:
    rows.append([cell(cc(f)), cell(k), csm(d)])
story.append(table(["File", "Version", "Contents / hook"], rows, [1.9*inch, 0.7*inch, 7.1*inch]))

frame_p = Frame(M, 0.7 * inch, PAGE_W - 2 * M, PAGE_H - 1.4 * inch, id="p")
frame_l = Frame(0.5 * inch, 0.65 * inch, L_W - 1.0 * inch, L_H - 1.3 * inch, id="l")
doc = BaseDocTemplate("/mnt/user-data/outputs/v4_4_grouping/DSO_v4_4_Score_Grammar_Companion.pdf", pagesize=letter,
                      pageTemplates=[PageTemplate(id="portrait", frames=[frame_p], onPage=bg_factory(PAGE_W, PAGE_H)),
                                     PageTemplate(id="wide", frames=[frame_l], onPage=bg_factory(L_W, L_H), pagesize=landscape(letter))],
                      title="DSO v4.3 — Topological Score Grammar: implementation companion", author="Sam")
doc.build(story)
print("done")
