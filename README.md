# Code-commenting
- Undergraduate Thesis on Code Comment Generation done under the guidance of [Prof. N. L. Bhanu Murthy](https://www.bits-pilani.ac.in/hyderabad/bhanumurthy/Profile)

# Dataset
- We use the dataset of the [DeepCom code](https://github.com/xing-hu/EMSE-DeepCom) for our training. We have trained mostly on Google Colaboratory. Recommend trying for the university's HPC (if not too busy) or GCloud credits (if **ANY** of your cards can get through without a refund).

# Results  

- Our model did not cross SOTA performance, which is something we have expected. It has however managed to produce semantically correct comments, occasionally more informative than the user's comments themselves. 
- Many of the comments within the first epoch were repetitive, but the number of meaningful comments increased significantly over time. 

NOTE: The first line is the comment spit out by the machine. The second one is the true human comment:  
![image](https://user-images.githubusercontent.com/39939017/116540577-f7ab5c80-a907-11eb-921b-7676bbc4b27e.png)

The model requires more training for rarer tokens:  
![image](https://user-images.githubusercontent.com/39939017/116540635-0eea4a00-a908-11eb-96c5-20c606e72b16.png)

Here the model fails to spit a grammatically correct word, but it can capture the inner semantics of the code:  
![image](https://user-images.githubusercontent.com/39939017/116540706-26c1ce00-a908-11eb-8fd7-39ba26fa0410.png)

Due to teacher-forcing, the machine has been confused, but otherwise, it still tried to produce a meaningful comment when humans gave a bad comment:  
![image](https://user-images.githubusercontent.com/39939017/116540815-448f3300-a908-11eb-8206-e8ee8e291f68.png)

The model can also substitute words with similar meanings:  
![image](https://user-images.githubusercontent.com/39939017/116540865-5c66b700-a908-11eb-84f7-bc94f4c7cac2.png)

![image](https://user-images.githubusercontent.com/39939017/116540912-68527900-a908-11eb-8494-b2769b46dd86.png)

## BLEU Scoring and Loss plots

![meteor](https://github.com/AetherPrior/Code-commenting/blob/beta-testing/Figure_3.png)

![bleu](https://github.com/AetherPrior/Code-commenting/blob/beta-testing/Figure_2.png)  

![loss](https://github.com/AetherPrior/Code-commenting/blob/beta-testing/Figure_1.png)  

## Box Plots

![Bleu](https://github.com/AetherPrior/Code-commenting/blob/beta-testing/BLEU4.png)

![METEOR](https://github.com/AetherPrior/Code-commenting/blob/beta-testing/METEOR.png)

# Contributing
- Please read [CONTRIBUTING.md](https://github.com/AetherPrior/Code-commenting/blob/main/CONTRIBUTING.md) for guidelines

