# We defaulty put all data in the Train/datasets. You can put them anywhere but create softlinks under Train/datasets.

# We provide two way to download data. 1) Cloudstor; 2) Google Drive
# 1. Download from CloudStor:

# download part-fore
cd Train/datasets
mkdir DiverseDepth
cd DiverseDepth
wget https://cloudstor.aarnet.edu.au/plus/s/HNfpS4tAz3NePtU/download -O DiverseDepth_d.tar.gz
wget https://cloudstor.aarnet.edu.au/plus/s/n5bOhKk52fXILp9/download -O DiverseDepth_rgb.zip
tar -xvf DiverseDepth_d.tar.gz
unzip DiverseDepth_rgb.zip


# download part_in, collected from taskonomy
cd ..
mkdir taskonomy
cd taskonomy
# (original link) wget https://cloudstor.aarnet.edu.au/plus/s/Q4jqXt2YfqcGZvK/download -O annotations.zip
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1Nhz0BABZBjjE-ITbJoSyVSuaJkD-iWVv" -O annotations.zip
wget https://cloudstor.aarnet.edu.au/plus/s/EBv6jRp326zMlf6/download -O taskonomy_rgbs.tar.gz
wget https://cloudstor.aarnet.edu.au/plus/s/t334giSOJtC97Uq/download -O taskonomy_ins_planes.tar.gz
wget https://cloudstor.aarnet.edu.au/plus/s/kvLcrVSWfOsERsI/download -O taskonomy_depths.tar.gz
tar -xvf ./*.tar.gz
unzip annotations.zip

# HRWSI
cd ..
mkdir HRWSI
cd HRWSI
wget https://cloudstor.aarnet.edu.au/plus/s/oaWj2Cfvif3WuD0/download -O HRWSI.zip
unzip HRWSI.zip

# Holopix50k
cd ..
mkdir Holopix50k
cd Holopix50k
wget https://cloudstor.aarnet.edu.au/plus/s/LuOsawtGq6cDAKr/download -O Holopix50k.zip
unzip Holopix50k.zip

# The overview of data under Train/datasets are:
# -Train
# |--datasets
#    |--DiverseDepth
#       |--annotations
#       |--depths
#       |--rgbs
#    |--taskonomy
#       |--annotations
#       |--depths
#       |--rgbs
#       |--ins_planes
#    |--HRWSI
#    |--Holopix50k
