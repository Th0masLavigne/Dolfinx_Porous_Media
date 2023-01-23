# Multi-compartment poroelastic models of perfused biological soft tissues: implementation in FEniCSx

This repository contains the codes used to generate the example presented in *Lavigne et al.*[^1]. Porous-media modelling of confined compression columns (with one or two fluid compartments) were implemented using the FEniCSx OpenSource software. The architecture of the repository follows the one of the corresponding article.

The version used of FEniCSx is v0.5.2. Pandas were integrated for the visualization (excel and csv). [Dockerfile](Dockerfile) and built images are made available. To pull the image, after having installed docker, run `th0maslavigne/dolfinx:v0.5.2`. Otherwise the image can be built by running `docker build .` in the folder of the DockerFile. Then the container can be interactively executed through the command below:

```sh
docker run -ti -v $(pwd):/home/fenicsx/shared -w /home/fenicsx/shared th0maslavigne/dolfinx:v0.5.2
```

All codes were made suitable for parallel computation (except for the convergence analysis). Parallel instance can be computed using:

```sh 
mpirun -n <N > python3 < filename >
```

Where <N> is the number of threads to use and <filename> is the python code of the problem.

A [.gitignore](.gitignore) has been created to avoid importing the heavy results.

##  Section 2: Confined compression of a column: geometrical definition

All the benchmark examples focus on uni-axial confined compression of a column sample as show below. Both 2D and 3D geometries are studied.

<img 
    style="display: block; 
           margin-left: auto;
           margin-right: auto;
           width: 30%;"
src=https://user-images.githubusercontent.com/91893320/214018911-5a95fb22-6066-4546-9399-afbe2c3b3cdc.png
alt="Geometry">
</img>

This section aims to provide the minimal working codes for the generation and boundary marking for the 2D and 3D domains (meshes). Further informations are provided [here](Section_2/readme.md).

## Section 3: Single-compartment porous medium

[Two benchmark examples](Section_3) were studied for a single-compartment (fluid) porous medium. First, a medium with an elastic scaffold (in 2D) was studied and compared to the analytical Terzaghi solution:

$$\text{p}=\frac{4p_0}{\pi}\sum_{k=1}^{+\infty}\frac{(-1)^{k-1}}{2k-1}\cos[(2k-1)\frac{\pi}{2}\frac{z}{h}]\exp[-(2k-1)^2\frac{\pi^2}{4}\frac{c_vt}{h^2}]$$

$$ c_v = \frac{k^\varepsilon}{\mu^l(S+\frac{\beta^2}{M})}$$

$$ M = \frac{3K^s(1-\nu)}{(1+\nu)}$$

$$ S = \frac{\beta-\varepsilon^l_0}{K^s} + \frac{\varepsilon^l_0}{K^l}$$

Second, the solid scaffold was considered hyper-elastic (in 3D to be compatible with the energy potential definitions). The solution was compared to *Selvadurai et al.*[^2] previous work.

The performances of the code were evaluated and compared to the former FEniCS software.

## Section 4: Confined bi-compartment porous-elastic medium

This folder contains a [bi-compartment](Section_4/readme.md) model of a vascularized model, based on *Sciumè et al.*[^3] previous work. Conversely to the Cast3m implementation (staggered computation), the proposed workflow uses a monolithic solving of the weak form. 

Both Minimal and complete working codes are provided.

## Appendices
### Apppendix A
The 3D Terzaghi example is provided [here](Appendix/appendix_a.py).

### Apppendix B
The example of local refinement using GMSH API, FEniCSx and their importation within the FEniCSx environment are respectively provided [here](Appendix/appendix_b1.py), [here](Appendix/appendix_b2.py) and [here](Appendix/appendix_b3.py).

### Appendix C
The example of the function used for evaluation is provided [here](Appendix/appendix_c.py)


[^1]: Titre, auteurs, doi
[^2]: Titre, auteurs, doi
[^3]: Titre, auteurs, doi
