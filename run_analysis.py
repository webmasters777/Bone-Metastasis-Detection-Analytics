import sys
sys.path.insert(0, '.')
from analysis import batch_classify_dataset, load_dataset_labels

print('='*70)
print('🦴 BONE METASTASIS DETECTION SYSTEM - COMPLETE OUTPUT')
print('='*70)
print()

# 1. Dataset Statistics
print('📊 DATASET STATISTICS')
print('-'*70)
try:
    labels_rant = load_dataset_labels('RANT')
    labels_rpost = load_dataset_labels('RPOST')

    rant_normal = sum(1 for v in labels_rant.values() if v == 0)
    rant_meta = sum(1 for v in labels_rant.values() if v == 1)
    rpost_normal = sum(1 for v in labels_rpost.values() if v == 0)
    rpost_meta = sum(1 for v in labels_rpost.values() if v == 1)

    print(f'chestRANT Dataset:')
    print(f'  ├─ Normal images:     {rant_normal}')
    print(f'  ├─ Metastasis images: {rant_meta}')
    print(f'  └─ Total:             {rant_normal + rant_meta}')
    print()
    print(f'chestRPOST Dataset:')
    print(f'  ├─ Normal images:     {rpost_normal}')
    print(f'  ├─ Metastasis images: {rpost_meta}')
    print(f'  └─ Total:             {rpost_normal + rpost_meta}')
    print()
    print(f'Combined (RANT + RPOST):')
    print(f'  ├─ Total Normal:      {rant_normal + rpost_normal}')
    print(f'  ├─ Total Metastasis:  {rant_meta + rpost_meta}')
    print(f'  └─ GRAND TOTAL:       {rant_normal + rant_meta + rpost_normal + rpost_meta}')
    print()
except Exception as e:
    print(f'Error loading dataset stats: {e}')

# 2. Test Results - RANT (10 images)
print('🎯 MODEL PERFORMANCE - chestRANT (First 10 Images)')
print('-'*70)
try:
    rant_results = batch_classify_dataset('RANT', limit=10)
    rant_perf = rant_results['performance']
    print(f'Total Tested: {rant_results["total_images"]}')
    print(f'Accuracy:     {rant_perf["accuracy"]:.1%}')
    print(f'Sensitivity:  {rant_perf["sensitivity"]:.1%} (metastasis detection)')
    print(f'Specificity:  {rant_perf["specificity"]:.1%} (normal detection)')
    print(f'Precision:    {rant_perf["precision"]:.1%}')
    print(f'F1-Score:     {rant_perf["f1_score"]:.4f}')
    print()
    print('Confusion Matrix:')
    print(f'  ├─ True Positives:  {rant_perf["true_positives"]}')
    print(f'  ├─ True Negatives:  {rant_perf["true_negatives"]}')
    print(f'  ├─ False Positives: {rant_perf["false_positives"]}')
    print(f'  └─ False Negatives: {rant_perf["false_negatives"]}')
    print()
except Exception as e:
    print(f'Error in RANT analysis: {e}')

# 3. Test Results - RPOST
print('🎯 MODEL PERFORMANCE - chestRPOST (First 10 Images)')
print('-'*70)
try:
    rpost_results = batch_classify_dataset('RPOST', limit=10)
    rpost_perf = rpost_results['performance']
    print(f'Total Tested: {rpost_results["total_images"]}')
    print(f'Accuracy:     {rpost_perf["accuracy"]:.1%}')
    print(f'Sensitivity:  {rpost_perf["sensitivity"]:.1%}')
    print(f'Specificity:  {rpost_perf["specificity"]:.1%}')
    print(f'Precision:    {rpost_perf["precision"]:.1%}')
    print(f'F1-Score:     {rpost_perf["f1_score"]:.4f}')
    print()
    print('Confusion Matrix:')
    print(f'  ├─ True Positives:  {rpost_perf["true_positives"]}')
    print(f'  ├─ True Negatives:  {rpost_perf["true_negatives"]}')
    print(f'  ├─ False Positives: {rpost_perf["false_positives"]}')
    print(f'  └─ False Negatives: {rpost_perf["false_negatives"]}')
    print()
except Exception as e:
    print(f'Error in RPOST analysis: {e}')

# 4. Combined Test
print('🎯 MODEL PERFORMANCE - Combined (RANT + RPOST, First 100 Each)')
print('-'*70)
try:
    combined_results = batch_classify_dataset('BOTH', limit=200)
    combined_perf = combined_results['performance']
    print(f'Total Tested: {combined_results["total_images"]}')
    print(f'Accuracy:     {combined_perf["accuracy"]:.1%}')
    print(f'Sensitivity:  {combined_perf["sensitivity"]:.1%}')
    print(f'Specificity:  {combined_perf["specificity"]:.1%}')
    print(f'Precision:    {combined_perf["precision"]:.1%}')
    print(f'F1-Score:     {combined_perf["f1_score"]:.4f}')
    print()
    print('Confusion Matrix:')
    print(f'  ├─ True Positives:  {combined_perf["true_positives"]}')
    print(f'  ├─ True Negatives:  {combined_perf["true_negatives"]}')
    print(f'  ├─ False Positives: {combined_perf["false_positives"]}')
    print(f'  └─ False Negatives: {combined_perf["false_negatives"]}')
    print()
except Exception as e:
    print(f'Error in combined analysis: {e}')

print('='*70)
print('✅ PROJECT EXECUTION COMPLETE')
print('='*70)
