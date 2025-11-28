# Frequently Asked Questions (FAQ)

## General Questions

### Q: What is this repository?
**A:** Complete replication package for our PLOS ONE paper on preventing temporal data leakage in CI/CD build prediction. Includes all code, data, and documentation to reproduce our findings.

### Q: What are the main contributions?
**A:**
1. **3-type temporal leakage taxonomy** (Type 1: Direct Outcome, Type 2: Execution-Dependent, Type 3: Future Information)
2. **Platform-dependent leakage finding**: 15.07pp tax on Travis CI vs 0.48pp on GitHub Actions
3. **Realistic performance expectations**: 82-83% accuracy (not inflated 95-99%)
4. **Open-source toolkit** for detecting leakage in any SE prediction task

### Q: How long does reproduction take?
**A:**
- **Quick test:** 30 minutes (verify it works)
- **Complete reproduction:** 7-8 hours (all results and figures)
- **Just training models:** 2-3 hours

### Q: Do I need GPUs?
**A:** No. Random Forest doesn't benefit from GPU. Multi-core CPU is sufficient (`n_jobs=-1` uses all cores).

---

## Technical Questions

### Q: Why only 82-83% accuracy instead of 95%+?
**A:** We use only **pre-build features** (no temporal leakage). Prior work reporting 95-99% included execution-dependent features (build duration, test counts) that are only available **after** the build completes - these provide perfect hindsight but zero prospective utility.

**Our 82-83% is realistic and deployable. Their 95%+ is retrospective and unachievable in practice.**

### Q: What's the difference between clean and leaky models?
**A:**
- **Clean model:** Uses only 31 pre-build features → 82.73% accuracy → Deployable in production
- **Leaky model:** Uses all 66 features including execution results → 97.80% accuracy → Only works in hindsight

The 15.07pp difference is the "leakage tax" - accuracy inflation from using unavailable information.

### Q: Why does GitHub Actions have minimal leakage (0.48pp) vs Travis CI (15.07pp)?
**A:** **Platform architecture matters!**
- GitHub Actions: Native integration with GitHub → accurate prediction from static features alone
- Travis CI: Third-party service → benefits substantially from time-dependent popularity metrics (stars, forks)

This is our most surprising finding - leakage vulnerability varies by platform.

### Q: Can I use this on my company's CI/CD data?
**A:** Yes! The leakage detection toolkit works on any CI/CD dataset:
```python
from src.leakage_detection import TemporalLeakageTaxonomy

taxonomy = TemporalLeakageTaxonomy()
clean_features = taxonomy.classify_features(your_data, target='build_success')
```

---

## Data Questions

### Q: Where are the original datasets?
**A:**
- **TravisTorrent:** https://travistorrent.testroots.org/ (DOI: 10.5281/zenodo.1254890) - 2.64M builds
- **GHALogs:** https://zenodo.org/record/10154920 (DOI: 10.5281/zenodo.10154920) - 513K workflows

**Our processed datasets** (100K TravisTorrent sample, 75.7K GHALogs enriched) are included in this package.

### Q: Why not use the full 2.64M TravisTorrent builds?
**A:** 100,000 builds exceed statistical power requirements (needed only 78,400 for 80% power, 5pp effect, α=0.05). Using 100K provides 28% power margin while keeping training time reasonable (<5 minutes).

### Q: What programming languages are included?
**A:**
- **TravisTorrent:** Java (40%), Ruby (35%), Python (15%), JavaScript (10%)
- **GHALogs:** Mixed (language field varies widely across 7,620 repos)

### Q: Are there any privacy concerns?
**A:** No. All data are from:
- Public GitHub repositories
- Public CI/CD build logs (Travis CI, GitHub Actions)
- No personally identifiable information
- No proprietary code or sensitive data

---

## Results Questions

### Q: My results don't match exactly. What's wrong?
**A:** Check:
1. **Random seed** - Must be 42 everywhere
2. **scikit-learn version** - Use 1.0+ (exact version: 1.3.2)
3. **Data preprocessing** - Applied leakage taxonomy? (31 features for TravisTorrent)
4. **Train/test split** - 80/20 with stratification

Small differences (±0.5%) are normal. Differences >1% indicate a problem.

### Q: Why is my leakage tax different?
**A:** Ensure you're comparing:
- **Clean model:** 31 features for TravisTorrent (check `travistorrent_leakage_taxonomy.csv`)
- **Leaky model:** ALL 66 features (including tr_duration, tr_status, etc.)

### Q: What if I get lower accuracy (e.g., 75%)?
**A:** Possible causes:
- Using wrong features (check you have all 31 clean features)
- Incorrect hyperparameters (max_depth should be 10, not 3)
- Data issues (missing values not imputed)
- Wrong model (should be Random Forest, not Decision Tree alone)

---

## Usage Questions

### Q: How do I run just one experiment?
**A:**
```bash
cd experiments/travistorrent
python RQ1_leakage_free_performance.py  # Just RQ1
```

### Q: How do I run all experiments?
**A:**
```bash
cd experiments/travistorrent
./run_all_travistorrent.sh  # All TravisTorrent RQs

cd ../ghalogs
./run_all_ghalogs.sh  # All GHALogs RQs

cd ../combined
python platform_comparison.py  # Cross-platform analysis
```

### Q: Can I modify the code?
**A:** Yes! MIT License - you can:
- ✅ Use for any purpose (research, commercial, educational)
- ✅ Modify and adapt
- ✅ Redistribute
- ✅ Create derivative works

**Only requirement:** Include original copyright notice and license.

### Q: How do I apply this to defect prediction / test selection / other tasks?
**A:** The taxonomy generalizes to any temporal SE prediction task:
1. Load your dataset
2. Apply `TemporalLeakageTaxonomy` to classify features
3. Remove Type 1, Type 2, Type 3 features
4. Train on clean features only
5. Measure leakage tax by comparing clean vs leaky performance

See `docs/LEAKAGE_TAXONOMY.md` section "Generalization to Other SE Prediction Tasks"

---

## Performance Questions

### Q: Can I speed up training?
**A:** Yes:
```python
# Use fewer trees (trade accuracy for speed)
RandomForestClassifier(n_estimators=50)  # Instead of 100

# Or reduce sample size
df_sample = df.sample(n=50000, random_state=42)

# Or use all CPU cores
RandomForestClassifier(n_jobs=-1)  # Default in our scripts
```

### Q: How much memory is needed?
**A:**
- **Minimum:** 8 GB (may struggle with full dataset)
- **Recommended:** 16 GB (comfortable)
- **Our setup:** 128 GB (overkill, but fast)

100K builds with 31 features requires ~2-3 GB peak memory.

---

## Contribution Questions

### Q: Can I contribute improvements?
**A:** Yes! Please:
1. Fork the repository
2. Make your changes
3. Add tests if applicable
4. Submit a pull request

We welcome:
- Bug fixes
- Documentation improvements
- New platform support (GitLab CI, CircleCI, etc.)
- Performance optimizations
- Additional visualizations

### Q: How do I cite this work?
**A:** See `CITATION.cff` file or README.md for BibTeX citation.

### Q: Can I use this in my thesis/dissertation?
**A:** Absolutely! That's exactly what we built this for. Please:
- ✅ Cite our PLOS ONE paper
- ✅ Cite the original datasets (TravisTorrent, GHALogs)
- ✅ Acknowledge the toolkit if you use it
- ✅ Share your results with us (we'd love to know!)

---

## Contact

**Still have questions?**

**Corresponding Author:** Lalit Narayan Mishra
- Email: lnm8910@gmail.com
- Affiliation: Lowe's Companies, Inc.

**Co-Authors:**
- Amit Rangari: amitrangari@gmail.com
- Sandesh Nagrare: sandesh.nagrare@vit.edu.in
- Saroj Kumar Nayak: saroj.nayak@cb.amrita.edu

---

**Last Updated:** November 2025
**Document Version:** 1.0
