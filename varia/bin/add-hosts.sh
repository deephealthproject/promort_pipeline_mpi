#!/bin/bash
: > ~/.ssh/known_hosts
# docker swarm
for l in `nslookup tasks.mpi | tail -n +5 | grep Ad | cut -c 10-`;
do
        ssh-keyscan $l >> ~/.ssh/known_hosts
done
# kubernetes
for l in `nslookup mpi.default.svc.cluster.local | tail -n +5 | grep Ad | cut -c 10-`;
do
        ssh-keyscan $l >> ~/.ssh/known_hosts
done
