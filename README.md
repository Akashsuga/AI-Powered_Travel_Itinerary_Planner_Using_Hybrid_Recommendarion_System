# Travel-Recommendation-System

## 🚀 Overview
A hybrid machine learning system that generates personalized travel recommendations across Canada using:

- Content-Based Filtering (CBF)
- Collaborative Filtering (SVD Matrix Factorization)
- Popularity-Based Ranking
- Constraint Filtering (budget, season, province)

The system is designed to handle both **existing users** and **cold-start users**, ensuring useful recommendations even with no prior data.

---

## 🎯 Key Features

- Hybrid recommendation engine  
- Cold-start user handling  
- Budget & seasonal filtering  
- FastAPI-based real-time API  
- Evaluation using ranking metrics  

---

## 🧠 Model Architecture

**Final Ensemble Formula:**

0.45 × CBF + 0.35 × CF + 0.20 × Popularity

### Components:
- **TF-IDF** for destination tags  
- **MinMax Scaling** for numerical features  
- **Truncated SVD** for collaborative filtering  
<img width="1408" height="768" alt="WhatsApp Image 2026-04-09 at 8 07 26 PM" src="https://github.com/user-attachments/assets/f4162982-2d64-406e-984a-1b9b7b38f20a" />

---

## 📊 Performance Metrics

| Metric        | Score  |
|--------------|--------|
| Precision@10 | 0.216  |
| Recall@10    | 0.648  |
| NDCG@10      | 0.467  |

---

## 📁 Project Structure

```
travel_rec_system/
├── API/              # FastAPI backend
├── Data Source/     # dataset (synthetic)
├── graphs/          # evaluation plots
├── Screenshots/     # screenshots of UI
├── src/             # ML pipeline
├── requirements.txt
├── TRS- PPT.pdf     # project Powerpoint Presentation
├── project report.pdf
└── demo.html
```

---

## ⚙️ How to Run


## ⚙️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python data/generate_dataset.py
```

---

### 3. ▶️ Run Options

### Option A: Run ML Pipeline
```bash
python src/recommender.py
```
- Demo recommendations  
- Evaluation metrics (Precision, Recall, NDCG, RMSE)

---

### Option B: Start API Server
```bash
uvicorn api.main:app --reload
```
Open: http://localhost:8000/docs



---

### 4. 🌐 Demo (No Setup)

Open in browser:
```
intelligent_travel_recommendation_system_canada.html
```
---

## 📊 Example Output

**User Input:**
- Budget: $200  
- Season: Winter  
- Province: Alberta 

**Top Recommendations:**
1. Jasper Dark Sky Preserve
2. Banff National Park 
3. Lake Louise
   
   <img width="1477" height="926" alt="Screenshot_7-7-2026_1275_" src="https://github.com/user-attachments/assets/cb7e7b64-8e32-49a7-806c-d092d4785f64" />


---

## 📈 Visual Insights

The project includes visual analysis of:

- Interaction matrix sparsity  
- Hybrid weight tuning  
- Seasonal demand trends  

(Refer to `/graphs` folder)


---

## ⚠️ Notes

- Uses synthetic dataset (academic purpose)  
- Built as an explainable ML prototype  

---

## 💡 Future Improvements

- Deploy API (Render / AWS)  
- Add real-world dataset  
- Build frontend interface  

---



## 👨‍💻 Author

**AKASH M**  
Data Science Student – Tamil Nadu  
- [GitHub](https://github.com/Akashsuga) |
- [LinkedIn](https://www.linkedin.com/in/akash-m-0b09602ba)

---
