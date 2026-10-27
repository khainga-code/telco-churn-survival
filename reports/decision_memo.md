# Decision memo: retention targeting and pricing

**To:** commercial lead, subscription business
**From:** data science
**Data:** IBM Telco Customer Churn snapshot, 7,043 customers, 26.5% churned
**Status:** analysis complete; economics are placeholders pending finance sign-off

---

## Recommendation

1. **Run a retention pilot on month-to-month fibre customers.** They combine the
   highest churn risk with the highest revenue at risk. Use the contact queue
   (`reports/contact_queue.csv`) rather than a pure risk ranking; at 500 contacts
   it is worth about £1,000 more under the stated assumptions.
2. **Do not reprice from this analysis.** A 10% price rise is associated with
   +2.0pp churn under one defensible adjustment set and -1.2pp under another. The
   sign is not identified from observational data.
3. **Size the pilot properly.** 1,494 customers per arm detects a 5pp reduction at
   80% power. If whole agent areas are randomised, plan for about 3,242 per arm
   (82 areas at roughly 40 customers each, ICC 0.03).

## The three numbers that matter

| Question | Answer | What it means |
|---|---|---|
| Can we rank risk well enough to act? | AUC **0.818** [0.794, 0.840]; top 20% captures **46%** [41, 50] of churners at 61% precision | Yes. The base rate is 27%, so the list is worth working. |
| What is the contact policy worth? | Contact above **11%** risk: 884 of 1,409 holdout customers, expected **£23.0k**, realised **£22.8k**. Contacting everyone: **£19.6k** | The policy beats a blanket campaign by about 17% under the assumed economics. |
| When do customers leave? | 12-month survival **70%** (month-to-month) vs **99%** (one-year); **6.5 months** of subscription lost per customer by month 24 | Timing the contact matters as much as choosing the customer. |

Supporting evidence:

- Calibration is good: top decile 68% predicted vs 67% observed; calibration slope
  0.92. A simple logistic model is not outperformed by gradient boosting
  (ΔAUC +0.005, 95% CI [-0.001, +0.011]), so the simpler model is the better
  operating choice.
- Fibre customers leave sooner than DSL even within contract strata (HR 4.35,
  95% CI [3.57, 5.30]). This is stable and points at service quality, not price.
- The proportional-hazards assumption fails for contract and price, so all hazard
  ratios should be read as time-averages; the contract-stratified model is the
  robustness check.

## The price question

| Adjustment set | Effect of +10% price | 95% interval |
|---|---|---|
| Contract only | **+2.04pp** churn | [+1.79, +2.29] |
| Contract + internet service | **-1.23pp** churn | [-1.76, -0.67] |

Price and fibre are correlated at 0.79, and the price variance inflation factor
rises from 1.0 to 5.4 when internet service is added. The E-value is 1.27-1.36:
an unmeasured factor associated with both price and churn by only ~1.3× could
explain the whole association. **This is a reason to test prices, not to set them.**

## What would change this recommendation

- **Any one of the three assumptions moving by 2×.** Doubling the contact cost to
  £20, halving the save rate to 15%, or halving the value saved to £150 all raise
  the cut-off from 11% to 22% and shrink the list from 884 to 650 customers. The
  campaign still pays under a doubled contact cost (about £15.5k on the holdout);
  halving the save rate leaves about £7.7k, which is a much thinner case for
  running it at all.
- **No movement in the pilot.** If a well-powered offer does not reduce churn, the
  model is ranking a risk that the business cannot act on.
- **A different value definition.** The analysis assumes value at risk is
  proportional to the monthly bill. If lifetime margin is dominated by tenure or
  add-ons, the queue changes.

## Caveats

- Economics (£300 saved, £10 contact, 30% save) are assumptions, not measurements.
- Data are a single observational snapshot from one business; nothing here is a
  claim about PAYG, solar or LPG customers.
- The monitoring year in notebook 06 is simulated. It demonstrates the trigger,
  not the true drift rate.
- The pilot design assumes no interference between customers. Area randomisation
  needs the larger sample quoted above.
