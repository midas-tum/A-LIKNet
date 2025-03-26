# A-LIKNet

**Attention incorporated network for sharing low-rank, image and k-space information during MR image reconstruction to achieve single breath-hold cardiac Cine imaging**

> Official Tensorflow implementation of our paper published in *Computerized Medical Imaging and Graphics (CMIG), Volume 120, March 2025*.

📄 **[Read the paper here](https://www.sciencedirect.com/science/article/pii/S0895611124001526)**  
✏️ **Authors**: Siying Xu, Kerstin Hammernik, Andreas Lingg, Jens Kübler, Patrick Krumm, Daniel Rueckert, Sergios Gatidis, Thomas Küstner

---

## 🔧 Overview

A-LIKNet is a novel deep learning framework for dynamic MR image reconstruction. It leverages **multi-domain attention mechanisms** to share features across **low-rank**, **image**, and **k-space** representations.

The core architecture includes:
- UNet with time-wise attention
- Local spatial-temporal low-rank modules
- K-space networks with coil-wise attention
- Iterative consistency and information sharing layers

<p align="center">
  <img src="pictures/architecture.jpg" alt="A-LIKNet Architecture" width="700"/>
</p>

---

## 📁 Project Structure

```bash
A-LIKNet/
├── data_pipeline/        # Data preprocessing and loading
├── model/                # Network architecture and training logic
├── pictures/             # Visualizations (network diagrams etc.)
├── utils/                # Utility functions (metrics, visualization)
└── README.md             # Project description and usage
```

---

## 📂 Dataset

This project uses an **in-vivo cardiac Cine MR dataset**, which cannot be publicly released due to institutional data sharing restrictions.

If you are interested in reproducing the results, please contact the authors for potential collaboration or use publicly available alternatives such as:

- [OCMR](https://www.ocmr.info/)
- [CMRxRecon](https://www.synapse.org/Synapse:syn51471091/wiki/622170)

---

## 📽️ Presentation

We presented **A-LIKNet** at the **2023 ISMRM & ISMRT Annual Meeting & Exhibition**.

🎞️ [▶ Watch the presentation here](https://archive.ismrm.org/2023/0819.html)

> 📝 This presentation was based on an earlier version of our work, submitted as an extended abstract to ISMRM 2023.  
> The current repository reflects the full version published in *Computerized Medical Imaging and Graphics (2025)*.

---

## 📚 Citation

If you use this code or find our work helpful, please cite:

```
@article{xu2025ALIKNet,
  title={Attention incorporated network for sharing low-rank, image and k-space information during MR image reconstruction to achieve single breath-hold cardiac Cine imaging},
  author={Xu, Siying and Hammernik, Kerstin and Lingg, Andreas and Kübler, Jens and Krumm, Patrick and Rueckert, Daniel and Gatidis, Sergios and Küstner, Thomas},
  journal={Computerized Medical Imaging and Graphics},
  volume={120},
  pages={102475},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.compmedimag.2024.102475}
}
```

---

## 📬 Contact

For questions or collaboration opportunities, feel free to reach out to Siying Xu at siying.xu@med.uni-tuebingen.de
