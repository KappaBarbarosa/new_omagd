# TODO List - VQ Graph Reconstructer

## High Priority

### Zero Vector Token Configuration
- **Issue**: `zero_vector_token_id` is currently hardcoded to 293 in `mask_predictor.py`
- **Problem**: During training, if we mask nodes with GT=293 (zero-vector token), the model learns to always predict 293 since it's the most common token for invisible nodes
- **Current Fix**: Hardcoded `self.zero_vector_token_id = 293` to exclude these from masking
- **Future Solution**: 
  - Auto-detect zero-vector token ID from tokenizer by encoding a zero tensor
  - Or make it a configurable parameter in `mask_predictor_config`
- **File**: `src/modules/graph_reconstructers/mask_predictor.py` line ~238

---

## Medium Priority

### Obs Processor Node Index Consistency
- **Issue**: In `obs_processor.py`, index definitions may be inconsistent with actual node ordering
- **Details**: 
  - `self.enemy_start_idx = 1` but actual order is `[self, ally, ally, enemy, enemy, enemy]`
  - Need to verify and fix if incorrect
- **File**: `src/modules/graph_reconstructers/obs_processor.py` line 24-25

---

## Low Priority

### Evaluation-Only Mode Enhancements
- Add support for saving evaluation results to file
- Add visualization of token predictions
