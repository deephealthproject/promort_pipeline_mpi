#docker pull dhealth/pylibs-toolkit:1.2.0.1-cudnn
docker build -t mobydick.crs4.it:5000/mpi -f Dockerfile . \
&& docker push mobydick.crs4.it:5000/mpi
