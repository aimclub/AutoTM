Before you start:
1. One should remember that there is 'help' command thats describes various possible operations:
   ```
   ./bin/fitnessctl help
   ```
   
2. All operations with 'fitnessctl' can be executed in a different namespaces. 
   One only needs to set 'KUBE_NAMESPACE' variable.
   ```
   KUBE_NAMESPACE=custom_namespace ./bin/fitnessctl deploy
   ```

To run kube-fitness app in production with mlflow one should:

1. Set up .kube/config the way it can access and manage remote cluster.

2. Build the app's wheel:
   ```
   ./bin/fitnessctl build-app
   ```

3. Build docker images and install them into the shared private repository
   `./bin/fitnessctl install-images`. To accomplish this operation successfully you should have the repository added to your docker settings.
   See the example of the settings below.
   
4. Create mlflow volumes and PVCs (we require to supply desired_namespace because of PVCs) by issung the command:
   ```
   kubectl -n <desired_namespace> apply -f deploy/mlflow-persistent-volumes.yaml 
   ```
   Do not forget that the directories required by volumes should exist beforehand.

5. Deploy mlflow:
   ```
   ./bin/fitnessctl create-mlflow
   ```
   
6. Generate production config either using 'gencfg' command with appropriate arguments or using 'prodcfg' alone:
   ```
   ./bin/fitnessctl prodcfg
   ```
   
7. Finally deploy kube-fitness app with the command:
   ```
   ./bin/fitnessctl create
   ```
   
   Alternatively, one may safely use 'recreate' command if there may be already another deployment:
   ```
   ./bin/fitnessctl recreate
   ```
   
8. To check if everything works fine, one may create a test client and verify that it is completed successfully:
   ```
   ./bin/fitnessctl recreate-client
   ```

NOTE: there is a shortcut for the above sequence (EXCEPT p.4 and p.5):
```
./bin/fitnessctl deploy
```

To deploy the kube-fitness app on a kubernetes one should do the following:
   
1. Build docker images and install them into the shared private repository
   `./bin/fitnessctl install-images`. To accomplish this operation successfully you should have the repository added to your docker settings.
   See the example of the settings below.
   
3. Build and install the wheel into the private PyPI registry (private is the name of the registry).
   ```
   python setup.py sdist bdist_wheel register -r private upload -r private
   ```
   To perform this operation you need to have a proper config at **~/.pypirc**
   See the example below.
   

4. Go to the remote server that has access to your kubernetes cluster and has *kubectl* installed. 
   Install **kube_fitness** wheel 
   ```
   pip3 install --trusted-host node2.bdcl --index http://node2.bdcl:30496 kube_fitness --user --upgrade --force-reinstall
   ```
   
   The wheel should be installed with **--user** setting as it puts necessary files to **$HOME/.local/share/kube-fitness-data** directory.
   Other options are recommended for second and further reinstalls.

5. Add the line `export PATH="$HOME/.local/bin:$PATH"` into your **~/.bash_profile** to be able to call **kube-fitnessctl** utility.  

6. Run `kube-fitnessctl gencfg <arguments>` to create depployment file for kubernetes.

7. Run `kube-fitnessctl create` or `kube-fitnessctl recreate` to deploy fitness workers on your cluster.

8. To access the app from your jupyter server, you should install the wheel as well using pip.

9. See example how to run the client in kube_fitness/test_app.py

.pypirc example config:
```
[distutils]
index-servers =
  private

[private]
repository: http://<host>:<port>/
username: <username>
password: <password>
```
   

Docker daemon settings (**/etc/docker/daemon.json**) to access a private registry
`"insecure-registries":["node2.bdcl:5000"]`
