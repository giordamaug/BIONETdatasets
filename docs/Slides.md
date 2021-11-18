---
marp: true
paginate: true
backgroundImage: url('https://marp.app/assets/hero-background.svg')
---

<style>
td {
  font-size: 10px
}
th {
  font-size: 10px
}
</style>

# BIONETDatasets

What is it?
- A repository of biological datasets in TUD format
- A set of python wrappers 
    + for using graphs in TUD format within the more popular graph deep learning linraries: DGL, PyTorch, Spektral, Grakel
- Publicly available at: [Github](https://github.com/giordamaug/BIONETdatasets)


---
# The TU Datasets format file

---
# Results: GIN-0

-  5-fold stratified cross validation
- learning rate: 0.001, 100 epochs, batch size: 10, 3 layers, 64 channels

|dataset|loss                         |acc   | batch | channels | epochs | elapsed |
|-------|-----------------------------|------|------|------|------|------|
|MUTAG  |0.51                         |0.85  | 10   | 64   | 100  |
|PROTEINS|0.59                        |0.72  | 10   | 64   | 100  |
|Mutagenicity    |   0.537 | 0.801 | 10   | 64   | 100  | 24:05
|ogbg-molbbbp|0.43                    |0.86  | 10   | 64   | 100  |
|ogbg-molbace|0.63                    |0.76  | 10   | 64   | 100  |

---

# Results: HGP-SL

- one-random split: 80% train, 10% val, 10% test
- learning rate: 0.001, 1000 epochs with self stop, batch size: 512, 3 layers, 128 channels, dropout 0.0



|dataset        |loss                    |acc   | batch | stopped at | struct learn | sparse att | pool_ratio | elapsed |
|---------------|------------------------|------|-------|------------|----|---|----|----|
|MUTAG          |0.389                   |0.850 | 32    | 322        | :x:  | :x: | 0.5 | 105s
|PROTEINS       |0.381                   |0.857 | 512   |            | :heavy_check_mark:  |:heavy_check_mark: | 0.5 |
|Mutagenicity       |0.456 | 0.807| 512   |  206     | :heavy_check_mark:  |:heavy_check_mark: | 0.8 | 7499
|ogbg-molbbbp   |0.45 | 0.829  | 512   | 571        |:heavy_check_mark:  |:heavy_check_mark: |0.5 |1344s
|ogbg-molbace   | 0.522                  |0.743 | 512   | 224      | :heavy_check_mark:  |:heavy_check_mark: |0.5 | 3403s

----

# Results: NETPRO2VEC

-  one-random split: 90% train, 10% test
- agg_by [1], cut_of [0.1], dimensions 512, encodew False, 
  epochs 400, extractor: [1], min_count 2, prob_type ["ndd"], 
  vertex_attribute "label"

|dataset| acc Trans.   | acc Induc. | elapsed |
|---|----|----|----|
|MUTAG  | 0.947  | 0.947   |    | 
|KIDNEY  | 0.967  | 0.833   |    | 
|PROTEINS| 0.732 | 0.714 |    | 
|Mutagenicity    |    |   |    |
|ogbg-molbbbp|   |    |    |
|ogbg-molbace| 0.7628  | 0.790   |    |

