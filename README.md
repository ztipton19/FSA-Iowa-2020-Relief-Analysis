# USDA Farm Service Agency (FSA) 2020 Relief Analysis – Iowa

### Overview
This project analyzes the **2020 USDA Farm Service Agency (FSA)** payment distributions in **Iowa**, a year shaped by the dual crises of the **COVID-19 pandemic** and the **U.S.–China trade war**.  
The goal of this analysis is to understand how crisis-driven relief programs compared to traditional FSA payments and how these emergency measures reshaped agricultural support across the state.

### Course Context

This project was completed over a two-week period for ISYS 51003: Data Analytics Fundamentals at the University of Arkansas.
It served as a practical application of data cleaning, visualization, and statistical analysis using Python and real-world USDA datasets.

---

### Repository Contents
| File | Description |
|------|--------------|
| `USDA_Analysis.pdf` | Full report containing introduction, methodology, results, and conclusions. |
| `combined_fsa_analysis.py` | Python script for data cleaning, transformation, and analysis. |
| `README.md` | Project overview and documentation. |

Data Source: [USDA FSA Payment Files][fsa-foia]

[fsa-foia]: https://www.fsa.usda.gov/tools/informational/freedom-information-act-foia/electronic-reading-room/frequently-requested/payment-files

---

### Key Findings
- Financial assistance to Iowa farmers in 2020 exceeded normal FSA disbursements by **approximately 2.8×**.  
- **77%** of all payments were classified as **crisis-related**, including **COVID-19** and **Trade War** programs.  
- Crisis payments reached **new recipients** and were **roughly twice the average amount** of traditional support.  
- All **99 Iowa counties** received crisis assistance, with **Sioux County** leading total disbursements.  

These findings highlight the scale of federal intervention required to stabilize Iowa’s agricultural economy during one of the most disruptive years in modern history.

---

### Tools and Libraries
This project was completed in **Python 3.9+** using the following packages:

- `pandas` – data cleaning and transformation  
- `numpy` – numerical computation  
- `matplotlib` – visualization and plotting  
- `seaborn` – statistical graphics  
- `scipy` – hypothesis testing and statistical analysis  
- `geopandas` – geospatial visualization and county-level mapping  

---
