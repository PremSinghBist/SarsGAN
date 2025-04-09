The codebase presents an implementation of the paper "Generative AI in the Advancement of Viral Therapeutics for Predicting and Targeting Immune-Evasive SARS-CoV-2 Mutations," published in the IEEE Journal of Biomedical and Health Informatics. The paper can be accessed at https://doi.org/10.1109/JBHI.2024.3432649.

This codebase features a GAN implementation designed to generate SARS-CoV-2 Spike protein sequences. It has been meticulously customized and fine-tuned to meet specific implementation requirements. The original codebase is credited to Repecka, D., Jauniskis, V., Karpus, L., et al., and can be accessed via the following link: https://doi.org/10.1038/s42256-021-00310-5, with the GitHub repository available at https://github.com/Biomatter-Designs/ProteinGAN.

Additionally, the implementation of the SARS-Escape model is available through the collaborative efforts of Prem Singh Bist, Hilal Tayara, and Kil To Chong. This model aids in the prediction of SARS-COV-2 escape variants and can be referenced in the article published in Briefings in Bioinformatics, Volume 24, Issue 3, May 2023, bbad140, accessible at the following DOI: https://doi.org/10.1093/bib/bbad140 , Prem Singh Bist et al., Briefings in Bioinformatics GitHub Repository: https://github.com/PremSinghBist/Sars-CoV-2-Escape-Model.git.

For research purposes, all three Escape datasets, including Greany, Baum, and the Validation dataset, are publicly available on Zenodo: doi: 10.5281/zenodo.7142638.
Furthermore, the newly trained model, enhanced with augmented sequences, are located within the "pretrained" directory for reference and utilization.

## Supplementary File: 
The supplementary file described in the paper is also available  in the data/ directory. Here the link here[Supplementary File](https://github.com/PremSinghBist/SarsGAN/blob/master/data/Supplementary%20File.pdf)



## Network Training:
Access the training and validation datasets at data/protein/train_data to initiate the network training process. This directory contains the necessary datasets for both training and validation, essential for ensuring the model's accuracy and effectiveness during the training phase.

To train the GAN network, utilize the train_gan.py script located in the src directory. This script facilitates the training process, allowing you to refine the GAN model.

## Generating Spike Sequences:

To generate Spike sequences, employ the generate.py script found in the src directory. This script empowers the generation of Spike protein sequences, offering flexibility and control over the output.






