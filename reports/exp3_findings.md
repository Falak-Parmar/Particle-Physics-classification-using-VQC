# Experiment 03: Data Encoding Strategies

## Objective
Evaluate the impact of different quantum state preparation methods on classification performance: Angle Encoding, Amplitude Encoding, and Data Re-uploading.

## Methodology
- **Features:** 2 (`m_bb`, `missing energy mag.`)
- **Optimizer:** Adam (LR 0.05).
- **Comparison:** Mapping physical features to rotation angles vs. state amplitudes vs. layered re-injection.

## Encoding Comparison
| Strategy | Observed Performance |
|---|---|
| **Angle Encoding** | **Most Consistent:** Provides a robust and stable mapping for physics features. |
| Amplitude Encoding | **Information Loss:** Normalization requirements can discard relative magnitude data critical for event separation. |
| **Data Re-uploading** | **Highest Potential:** Increases model non-linearity by re-injecting features at every layer. |

## Key Insights
1. **Representational Power:** Data Re-uploading is the most effective architectural choice for maximizing the expressivity of shallow circuits.
2. **Feature Mapping:** Standard Angle encoding remains a highly effective and efficient baseline for most particle physics tasks.

---
*Reference Notebook:* `notebook/03_encoding_strategies.ipynb`
