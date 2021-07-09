# AutoTM
Automatic parameters selection for topic models (ARTM approach) using evolutionary algorithms

How to run experiments on kubernetes.

NOTE: If everything has already been setup earlier, 
you only need to execute only, p.3, p.4, p. 6 and p. 7 actions.
It all can be done with a single special command:
```
./bin/recreate-irace.sh
```

1. It is assumed that kube-fitness is already deployed on the cluster 
   in the right namespace.
   
2. One needs to copy fresh kube-fitness wheel in to the root of the project. 
   For example:
   ```
   cp ../kube-distributed-fitness/dist/kube_fitness-0.1.0-py3-none-any.whl .
   ```
   
3. Build the image of irace runner by issuing the command:
   ```
   ./bin/build-docker.sh
   ```

4. Push the image into the registry to make it accessible from anywhere in the cluster:
   ```
   ./bin/push-docker-to-repo.sh
   ``` 

5. Create pv and pvc to store results of irace:
    ```
    kubectl apply -f conf/irace-files-pvc-pv.yaml
   ```
   
6. Stop previous of irace if it exists (but it really shouldn't if one is doing it first time):
    ```
    ./bin/stop-irace.sh
   ```
   
7. Start a new instance of irace runner:
    ```
    ./bin/start-irace.sh
   ```
