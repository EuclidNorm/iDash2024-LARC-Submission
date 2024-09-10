from flamby.datasets.fed_tcga_brca import FedTcgaBrca

# To load the first center as a pytorch dataset
center0 = FedTcgaBrca(center = 0)
print(center0.data)
