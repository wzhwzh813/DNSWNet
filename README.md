# DNSWNet
Multi-Weather Image Restoration Network Based on Swin Transformer and CNN

About the Model:
The overall model architecture is shown below. This network draws inspiration from the concept of learning noise information in DNCNNs. By integrating the Swin Transformer Block with CNN, it effectively achieves an organic combination of locality and globality. This enables it to handle weather particle noise at different scales effectively, ultimately realizing a unified multi-weather degraded image restoration network.

<img width="1210" height="643" alt="图片" src="https://github.com/user-attachments/assets/c5560c18-82ba-4ec1-a987-06ee5fc80332" />





Visualization Results (Rain, Snow, and Fog Conditions):
从左往右依次为Input, MPRNet, SwinIR, RSFormer, DNSWNet(ours)


![rain_snow](https://github.com/user-attachments/assets/0484465b-64ef-4499-b6a5-ac0dc5bfea20)
![fog](https://github.com/user-attachments/assets/f1a47ab5-6c30-4beb-83c6-d9fc91846b50)




Quantitative analysis results：



<img width="754" height="308" alt="图片" src="https://github.com/user-attachments/assets/a6830859-5822-43ea-8cda-ab8dfb681611" />
<img width="750" height="303" alt="图片" src="https://github.com/user-attachments/assets/cdafdcdf-3892-42cf-a63e-d7c104cfeeaf" />
<img width="747" height="560" alt="图片" src="https://github.com/user-attachments/assets/9ba48763-2106-4174-8ea6-6af34a8f0f7b" />

