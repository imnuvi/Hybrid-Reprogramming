
To run the pipeline on demo data


The following code is setup to run on Slurm with singularity containers

```
git clone git@github.com:imnuvi/Hybrid-Reprogramming.git
cd Hybrid-Reprogramming/Genexpression/Epi2mePipeline
wget -P ./data https://ont-exd-int-s3-euwst1-epi2me-labs.s3.amazonaws.com/wf-single-cell/wf-single-cell-demo.tar.gz
tar -xzvf ./data/wf-single-cell-demo.tar.gz -C ./data
sh Epi2MeRunner.sh epi2me.conf
```

To run this with slurm, change the account and the notification mail id.

```
git clone git@github.com:imnuvi/Hybrid-Reprogramming.git
cd Hybrid-Reprogramming/Genexpression/Epi2mePipeline
wget -P ./data https://ont-exd-int-s3-euwst1-epi2me-labs.s3.amazonaws.com/wf-single-cell/wf-single-cell-demo.tar.gz
tar -xzvf ./data/wf-single-cell-demo.tar.gz -C ./data
sbatch epi2me.sbat
```
