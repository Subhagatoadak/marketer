# 🎨 **Marketer: Your AI-Powered Creative Content Generator**

[![Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-red)](https://streamlit.io/) [![Stability AI](https://img.shields.io/badge/API-Stability%20AI-blue)](https://stability.ai/) [![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)

Marketer is a cutting-edge AI-powered application designed to help creative professionals effortlessly generate and edit stunning marketing visuals. Built with Streamlit and integrated with Stability AI, Marketer enables rapid image creation, advanced editing, and customizable text overlays, empowering you to bring your marketing campaigns to life in minutes.

---

## 🌟 **Key Features**

### 🚀 **AI-Driven Image Generation & Editing**

Marketer harnesses powerful AI capabilities via Stability AI, offering:

- **Marketing Ad Generation:** Create professional-quality marketing ads instantly.
- **Control Sketch & Structure:** Generate highly customized images based on sketches or structural outlines.
- **Search and Recolor:** Easily recolor specific elements within your visuals.
- **Search and Replace:** Replace image components based on intelligent search prompts.
- **Background Replacement & Relighting:** Automatically swap backgrounds and adjust lighting for realistic results.
- **Creative Upscaling:** Enhance your images creatively, giving them a vibrant, professional look.

### 🖌️ **Dynamic Text Overlay**

- Overlay custom text on your visuals with adjustable color.
- Font size control is currently under improvement and will be enhanced in future updates.

### 🔄 **Flexible Image Reusability**

- Quickly reuse previously generated images for further edits and transformations.

### 💾 **Easy Download & Export**

- Effortlessly download your finalized images directly from the app, complete with overlays and edits.

---

## ⚙️ **Installation**

### 📋 **Prerequisites**
- Python 3.8 or later
- Stability AI API Key

### 🛠️ **Setup Instructions**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/marketer.git
   cd marketer
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Your Stability AI Key:**
   Create `.env`:
   ```bash
   STABILITY_KEY = "your-stability-ai-key"
   ```

5. **Run Marketer App:**
   ```bash
   streamlit run app.py
   ```

### 🐳 **Docker Deployment**

```bash
docker-compose up --build
```
Visit [http://localhost:8501](http://localhost:8501)

---

## 🚦 **How to Use**

### 1️⃣ **Select Task:**
- Choose from the sidebar: Marketing Ad, Control Sketch, Control Structure, Search and Recolor, Search and Replace, Replace Background & Relight, or Upscale Creative.

### 2️⃣ **Customize Inputs:**
- Provide prompts, seeds, and other task-specific settings.
- Optionally overlay text (font size control improvements coming soon).

### 3️⃣ **Generate & Download:**
- Generate your visuals and download them immediately for use.

---

## 📈 **Future Enhancements (Coming Soon)**

### 🎯 **Advanced Editing**
- Enhanced control over AI image generation parameters (temperature, randomness, etc.).
- Precision inpainting and fine-detail editing tools.

### 🎨 **Overlay Text Improvements**
- Comprehensive font size control and additional font styles.
- Text animations and custom text effects (shadow, outlines, gradients).

### 💬 **Collaborative Features**
- Real-time collaborative editing and feedback.
- Integrated project management for team collaboration.

### 📊 **Analytics & A/B Testing**
- Performance tracking of generated content.
- Built-in tools for A/B testing different visuals.

### 🔗 **Platform Integrations**
- Direct integrations with social media platforms and CMS for seamless content distribution.

### 🔐 **Enhanced Security**
- User authentication, roles, and secure data handling.

---

## 🤝 **Contributing**

We welcome contributions! Please:
- Fork the repository
- Submit pull requests or open issues
- Join our community discussions

---

## 📝 **License**

Licensed under the MIT License. 

---

© 2025 Marketer | Created by Subhagato Adak 🌟