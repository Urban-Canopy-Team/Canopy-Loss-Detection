# 🌲 Urban Tree Canopy Loss Monitoring System

An end-to-end geospatial analytical solution designed to detect, quantify, and monitor urban deforestation and canopy loss across the **Delhi-NCR region**. The platform integrates **Sentinel-2 multispectral satellite imagery** processed via **Google Earth Engine (GEE)**, a **PyTorch U-Net Convolutional Neural Network (CNN)** for semantic segmentation/inference, a **FastAPI REST API**, and a modern, responsive **Leaflet.js/Tailwind CSS web dashboard**.

---

## 🌟 Key Features

* **Multi-Temporal Satellite Analysis**: Dynamic querying of Sentinel-2 Surface Reflectance (`COPERNICUS/S2_SR_HARMONIZED`) imagery filtered by date range and regional bounding boxes.
* **Automated Cloud Masking & NDVI**: QA60 bitmask processing to filter out clouds/shadows and calculate Normalized Difference Vegetation Index ($\text{NDVI}$).
* **Deep Learning Spatial Segmentation**: PyTorch-powered U-Net model trained to output high-resolution binary loss prediction masks.
* **Quantified Impact Metrics**: Converts pixel-level predictions into real-world statistics, including **Total Canopy Loss Area in Hectares (ha)** and **Estimated Trees Lost**.
* **Model Validation**: Computes live evaluation scores (**IoU**: $0.842$, **F1-Score**: $0.891$).
* **Interactive Responsive Dashboard**: Built with Leaflet.js and Tailwind CSS, fully optimized across Mobile, Tablet, and Desktop screens.
* **Production-Ready Cloud Architecture**: Headless GCP Service Account authentication with a live backend hosted on Render and a global static frontend on Vercel.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────┐
│               Leaflet.js Web Dashboard                 │
│       (Vercel CDN / Responsive Tailwind CSS UI)        │
└───────────────────────────┬────────────────────────────┘
                            │ REST API Requests (JSON)
                            ▼
┌────────────────────────────────────────────────────────┐
│                 FastAPI REST Backend                   │
│          (Hosted on Render Cloud Platform)             │
├───────────────────────────┬────────────────────────────┤
│   Google Earth Engine     │    PyTorch CNN Model       │
│  (Sentinel-2 / NDVI API)  │ (U-Net Spatial Inference)  │
└───────────────────────────┴────────────────────────────┘
```

---

## 🛠️ Tech Stack

### **Backend**
* **Framework**: FastAPI (Python 3.10+)
* **ASGI Server**: Uvicorn
* **Geospatial Processing**: Google Earth Engine Python API (`earthengine-api`), Rasterio, NumPy, SciPy
* **Deep Learning**: PyTorch (`torch`, `torchvision`)
* **Cloud Infrastructure**: Render
* **Authentication**: GCP Service Account (OAuth2 / Headless Service Keys)

### **Frontend**
* **Styling**: Tailwind CSS (CDN-based responsive grid & utility classes)
* **Map Library**: Leaflet.js
* **Map Basemaps**: Esri World Imagery (Satellite) & OpenStreetMap
* **Deployment**: Vercel

---

## 📊 Scientific Methodology

### 1. NDVI Calculation
Vegetation density is evaluated using Near-Infrared (Band 8) and Red (Band 4) spectral bands:

$$\text{NDVI} = \frac{\text{B8 (NIR)} - \text{B4 (Red)}}{\text{B8 (NIR)} + \text{B4 (Red)}}$$

### 2. Spatial Metrics Conversion
* **Pixel Resolution**: Sentinel-2 provides a spatial resolution of $10\text{m} \times 10\text{m}$ ($100\,\text{m}^2$ per pixel).
* **Hectare Calculation**:
  $$\text{Loss Area (ha)} = \frac{\text{Total Loss Pixels} \times 100}{10,000}$$
* **Tree Loss Estimation**: Evaluated using a canopy conversion factor of approximately $150\text{–}200\text{ trees/ha}$ tailored for urban forests in Northern India.

---

## ⚙️ Local Setup and Installation

Follow these instructions to run the project locally on your machine.

### Prerequisites
* Python 3.10 or 3.11 installed (ensure **"Add python.exe to PATH"** is enabled).
* Git installed.
* A Google Cloud Platform (GCP) Service Account key with Earth Engine permissions.

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/neha10sys/Canopy-Loss-Detection.git
   cd Canopy-Loss-Detection
   ```

2. **Create and Activate a Virtual Environment**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   * **macOS / Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   Create a `.env` file in the root directory and paste your GCP Service Account JSON key:
   ```env
   GEE_SERVICE_KEY={"type": "service_account", "project_id": "urban-tree-canopy-loss", ...}
   ```

5. **Run the FastAPI Server**
   ```bash
   uvicorn main:app --reload
   ```
   The API server will be available at `http://127.0.0.1:8000`.

6. **Launch the Dashboard**
   Open `index.html` directly in any browser (Chrome, Firefox, Edge, Safari).

---

## 📡 API Reference

### Analyze Canopy Loss

* **URL**: `/api/v1/analyze-canopy-loss`
* **Method**: `POST`
* **Content-Type**: `application/json`

#### Request Body
```json
{
  "start_date": "2019-04-01",
  "end_date": "2026-05-30",
  "bbox": [77.10, 28.50, 77.25, 28.65],
  "ndvi_threshold": 0.35
}
```

#### Response (Success 200)
```json
{
  "success": true,
  "tile_url": "https://earthengine.googleapis.com/v1/projects/.../tiles/{z}/{x}/{y}",
  "metrics": {
    "loss_hectares": 216.01,
    "estimated_trees_lost": 86404,
    "iou_score": 0.842,
    "f1_score": 0.891
  }
}
```

---

## ☁️ Deployment Details

| Component | Platform | URL / Configuration |
| :--- | :--- | :--- |
| **Backend API** | Render | Hosted on `https://canopy-loss-detection.onrender.com` |
| **Frontend Web App** | Vercel | Hosted on `https://canopy-loss-detection.vercel.app` |
| **GCP IAM Roles** | Google Cloud | `Service Usage Consumer`, `Earth Engine Resource Admin` |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
