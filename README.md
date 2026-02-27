# 🚀 Predictive Inventory & Procurement Intelligence Platform

> **A web-based intelligent inventory planning platform for Manufacturing MSMEs that converts unstructured Excel-based operational data into predictive procurement intelligence.**

## 🧠 Problem Statement

Manufacturing MSMEs rely heavily on Excel-based sales, purchase, and stock records. This unstructured approach makes it difficult to anticipate anchor customer reorders, forecast raw material requirements, and maintain optimal inventory. The result is a reactive planning system leading to stockouts, excess inventory, and cash flow inefficiencies.

## 🎯 Core Features (MVP)

- **Smart Excel Data Ingestion**: Upload raw sales and stock register files with automatic formatting, cleaning, and structuring—no predefined format required.
- **Inventory Visibility Engine**: Tracks stock movement, computes current levels, and identifies fast/slow-moving SKUs with timeline visualizations.
- **Anchor Customer Demand Prediction**: Identifies revenue drivers, calculates reorder intervals, and predicts expected reorder windows with a confidence score.
- **BOM-Based Raw Material Forecasting**: Maps finished goods to raw materials via uploaded Bill of Materials (BOM), converting predicted demand into raw material requirements and providing procurement advisories.

## 🏗️ System Architecture

- **Frontend**: React + Tailwind CSS
- **Backend**: FastAPI (Python)
- **Data Processing**: Pandas, NumPy
- **Database**: PostgreSQL

**Workflow**:
`Excel Upload` → `Data Cleaning` → `Demand Modeling` → `Prediction Engine` → `Procurement Insight Dashboard`

## ⚙️ Setup Instructions

*(Dependencies and local execution steps will be added here as the project evolves)*

## 📈 Future Scope

- Supplier intelligence scoring
- Monte Carlo risk simulation
- Multi-warehouse optimization
- Price trend analysis
- Automated purchase order recommendation
