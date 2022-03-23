#! /bin/bash
: > hostfile
# docker swarm
for l in `nslookup tasks.mpi | tail -n +5 | grep Ad | cut -c 10-`;
do
	echo $l slots=1 >> hostfile
done
# kubernetes
for l in `nslookup mpi.default.svc.cluster.local | tail -n +5 | grep Ad | cut -c 10-`;
do
	echo $l slots=1 >> hostfile
done
