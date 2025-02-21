# DyAb-clone
DyAb reimplemented, see original [paper](https://www.biorxiv.org/content/10.1101/2025.01.28.635353v1.full.pdf) and [code](https://github.com/prescient-design/lobster). I also wrote some of my personal takeaways [here](https://tricky-art-e0d.notion.site/DyAb-sequence-based-antibody-design-and-property-prediction-in-a-low-data-regime-190a8b7648f1809ba211eee77f6d5b04).

## Brief Description
Since we're not doing the genetic algorithm portion of DyAb, it's simpler to pre-compute all of the embeddings for the antibody sequences as a data processing step. So the current setup expects a python dictionary where the key is the antibody id and the value is the [len, dim] embedding for that antibody.

Instead of antiBERTy or LBSTR, I tried [AbLang2](https://github.com/TobiasHeOl/AbLang2) to embed the antibody sequences. We can switch this out for other antibody language models easily.

I also aligned all the antibody sequences so they are all the same lengths, using gaps with [AntPack](https://github.com/jlparkI/AntPack). Since they are all the same length, the element-wise substraction step is now straightforward.

## Summary of DyAb
For each pair of antibodies, we predict the difference in binding affinity, or any other property of interest. To represent the pair of antibodies, we take the difference between embedding of antibody A and antibody B. As mentioned above, since they are both [len,dim], where the length is constant, the difference is also [len, dim].

A ResNet18 is the core model and expects a [batch, channel=3, dim, dim] input, so we use torch.vision's Resize to convert each [len,dim] difference representation into a square [img_dim, img_dim] representation. Since images usually have 3 channels (RGB), the original DyAb code seems to also use the addition or multiplication between the two embeddings as other channels. In my case, I just use the difference as the first channel and set the values of the other two channels to zeros. Lastly, the last ResNet.fc layer outputs the predicted difference.




