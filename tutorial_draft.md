tutorial\_draft
================

Evaluating Hyperparameter Tuning
================================

As mentioned in the Tuning tutorial, tuning a machine learning algorithm typically involves:

-   the hyperparameter search space, mention mlr objects for each bullet point
-   the optimization algorithm,
-   an evaluation method, i.e., a resampling strategy and a performance measure

After tuning, you may want to evaluate the tuning process in order to answer questions such as:

-   How does varying the value of a hyperparameter change the performance of the machine learning algorithm?
-   What's the most important hyperparameter?
-   Did the optimization algorithm (prematurely) converge?
-   When evaluating multiple hyperparameters, what is the contribution of each hyperparameter to performance?

mlr provides methods to generate and plot the data in order to evaluate the effect of hyperparameter tuning

Generating hyperparameter tuning data
-------------------------------------

mlr separates the generation of the data from the plotting of the data in case the user wishes to use the data in a custom way downstream.

The `generateHyperParsEffectData` method takes the tuning result along with 2 additional arguments: trafo and include.diagnostics. The trafo arg will convert the hyperparameter data to be on the transformed scale in case a trafo was used when creating the parameter (as in the case below). The include.diagnostics arg will tell mlr whether to include the eol and any error messages from the learner.

In the example below, we perform random search on the C parameter for SVM on the famous Pima Indians dataset. We generate the hyperparameter effect data so that the C parameter is on the transformed scale and we do not include diagnostic data.

``` r
ps = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x)
)
ctrl = makeTuneControlRandom(maxit = 100L)
rdesc = makeResampleDesc("CV", iters = 2L)
res = tuneParams("classif.ksvm", task = pid.task, control = ctrl, 
           measures = list(acc, mmce), resampling = rdesc, par.set = ps, 
           show.info = F)
generateHyperParsEffectData(res, trafo = T, include.diagnostics = F)
```

    ## HyperParsEffectData:
    ## Hyperparameters: C
    ## Measures: acc.test.mean,mmce.test.mean
    ## Optimizer: TuneControlRandom
    ## Nested CV Used: FALSE
    ## Snapshot of $data:
    ##             C acc.test.mean mmce.test.mean iteration
    ## 1  4.85077003     0.7656250      0.2343750         1
    ## 2  7.32314745     0.7604167      0.2395833         2
    ## 3 11.44262097     0.7539062      0.2460938         3
    ## 4  0.05774006     0.6510417      0.3489583         4
    ## 5  0.13913479     0.7330729      0.2669271         5
    ## 6  0.12853431     0.7135417      0.2864583         6

In the example below, we perform grid search on the C parameter for SVM on the famous Pima Indians dataset using nested cross validation. We generate the hyperparameter effect data so that the C parameter is on the untransformed scale and we do not include diagnostic data. As you can see below, nested cross validation is supported without any extra work by the user, allowing the user to obtain an unbiased estimator for the performance.

``` r
ps = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x)
)
ctrl = makeTuneControlGrid()
rdesc = makeResampleDesc("CV", iters = 2L)
lrn = makeTuneWrapper("classif.ksvm", control = ctrl, 
           measures = list(acc, mmce), resampling = rdesc, par.set = ps, 
           show.info = F)
res = resample(lrn, task = pid.task, resampling = cv2, extract = getTuneResult)
generateHyperParsEffectData(res)
```

    ## HyperParsEffectData:
    ## Hyperparameters: C
    ## Measures: acc.test.mean,mmce.test.mean
    ## Optimizer: TuneControlGrid
    ## Nested CV Used: TRUE
    ## Snapshot of $data:
    ##            C acc.test.mean mmce.test.mean iteration exec.time
    ## 1 -5.0000000     0.6875000      0.3125000         1     0.025
    ## 2 -3.8888889     0.6875000      0.3125000         2     0.025
    ## 3 -2.7777778     0.6875000      0.3125000         3     0.025
    ## 4 -1.6666667     0.7135417      0.2864583         4     0.025
    ## 5 -0.5555556     0.7656250      0.2343750         5     0.025
    ## 6  0.5555556     0.7500000      0.2500000         6     0.025
    ##   nested_cv_run
    ## 1             1
    ## 2             1
    ## 3             1
    ## 4             1
    ## 5             1
    ## 6             1

After generating the hyperparameter effect data, the next step is to visualize it. mlr has several methods built-in to visualize the data, meant to support the needs of the researcher and the engineer in industry. The next few sections will walk through the visualization support for several usecases.

Visualizing the effect of a single hyperparameter
-------------------------------------------------

In a situation when the user is tuning a single hyperparameter for a learner, the user may wish to plot the performance of the learner against the values of the hyperparameter.

In the example below, we tune the number of clusters against the silhouette
score on the Pima dataset. We specify the x-axis with the x arg and the y-axis with the y arg. If the plot.type arg is not specified, mlr will attempt to plot a scatterplot.

``` r
ps = makeParamSet(
  makeDiscreteParam("centers", values = 2:15)
)
ctrl = makeTuneControlGrid()
rdesc = makeResampleDesc("Holdout")
res = tuneParams("cluster.kmeans", task = mtcars.task, control = ctrl, 
           measures = silhouette, resampling = rdesc, par.set = ps, 
           show.info = F)
data = generateHyperParsEffectData(res)
plotHyperParsEffect(data, x = "centers", y = "silhouette.test.mean")
```

![](tutorial_draft_files/figure-markdown_github/single_cluster-1.png)

In the example below, we tune SVM with the C hyperparameter on the Pima dataset. We will use simulated annealing optimizer, so we are interested in seeing if the optimization algorithm actually improves with iterations. By default, mlr only plots improvements to the global optimum.

``` r
ps = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x)
)
ctrl = makeTuneControlGenSA(budget = 100L)
rdesc = makeResampleDesc("Holdout")
res = tuneParams("classif.ksvm", task = pid.task, control = ctrl, 
                 resampling = rdesc, par.set = ps, show.info = F)
data = generateHyperParsEffectData(res)
plotHyperParsEffect(data, x = "iteration", y = "mmce.test.mean",
               plot.type = "line")
```

![](tutorial_draft_files/figure-markdown_github/single_iters-1.png)

In the case of a learner crash, mlr will impute the crash with the worst value graphically and indicate the point. In the example below, we give the C parameter negative values, which will result in a learner crash for SVM.

``` r
ps = makeParamSet(
  makeDiscreteParam("C", values = c(-1, -0.5, 0.5, 1, 1.5))
)
ctrl = makeTuneControlGrid()
rdesc = makeResampleDesc("CV", iters = 2L)
res = tuneParams("classif.ksvm", task = pid.task, control = ctrl, 
           measures = list(acc, mmce), resampling = rdesc, par.set = ps, 
           show.info = F)
data = generateHyperParsEffectData(res)
plotHyperParsEffect(data, x = "C", y = "acc.test.mean")
```

![](tutorial_draft_files/figure-markdown_github/single_crash-1.png)

The example below uses nested cross validation with an outer loop of 2 runs. mlr indicates each run within the visualization.

``` r
ps = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x)
)
ctrl = makeTuneControlGrid()
rdesc = makeResampleDesc("Holdout")
lrn = makeTuneWrapper("classif.ksvm", control = ctrl, 
           measures = list(acc, mmce), resampling = rdesc, par.set = ps, 
           show.info = F)
res = resample(lrn, task = pid.task, resampling = cv2, extract = getTuneResult)
data = generateHyperParsEffectData(res)
plotHyperParsEffect(data, x = "C", y = "acc.test.mean", 
               plot.type = "line")
```

![](tutorial_draft_files/figure-markdown_github/single_nested-1.png)

Visualizing the effect of 2 hyperparameters
-------------------------------------------

In the case of tuning 2 hyperparameters simultaneously, mlr provides the ability to plot a heatmap and contour plot in addition to a scatterplot or line.

In the example below, we tune the C and sigma parameters for SVM on the Pima dataset. We use interpolation to produce a regular grid for plotting the heatmap. The interpolation arg accepts any regression learner from mlr to perform the interpolation. The z arg will be used to fill the heatmap or color lines, depending on the plot.type used.

``` r
ps = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -5, upper = 5, trafo = function(x) 2^x))
ctrl = makeTuneControlRandom(maxit = 100L)
rdesc = makeResampleDesc("Holdout")
learn = makeLearner("classif.ksvm", par.vals = list(kernel = "rbfdot"))
res = tuneParams(learn, task = pid.task, control = ctrl, measures = acc,
                 resampling = rdesc, par.set = ps, show.info = F)
data = generateHyperParsEffectData(res)
plotHyperParsEffect(data, x = "C", y = "sigma", z = "acc.test.mean",
                    plot.type = "heatmap", interpolate = TRUE)
```

![](tutorial_draft_files/figure-markdown_github/two-1.png)

We can use the show.experiments arg in order to visualize which points were specifically passed to the learner in the original experiment and which points were interpolated by mlr:

``` r
plotHyperParsEffect(data, x = "C", y = "sigma", z = "acc.test.mean",
                    plot.type = "heatmap", interpolate = TRUE, 
                    show.experiments = TRUE)
```

![](tutorial_draft_files/figure-markdown_github/two_show-1.png)

We can also visualize how long the optimizer takes to reach an optima for the same example:

``` r
plotHyperParsEffect(data, x = "iteration", y = "acc.test.mean", 
                    plot.type = "line")
```

![](tutorial_draft_files/figure-markdown_github/two_iters-1.png)

In the case where we are tuning 2 hyperparameters and we have a learner crash, mlr will indicate the respective points and impute them with the worst value. In the example below, we tune C and sigma, forcing C to be negative for some instances which will crash SVM. We perform interpolation to get a regular grid in order to plot a heatmap. We can see that the interpolation creates axis parallel lines resulting from the learner crashes.

``` r
ps = makeParamSet(
  makeDiscreteParam("C", values = c(-1, 0.5, 1.5, 1, 0.2, 0.3, 0.4, 5)),
  makeDiscreteParam("sigma", values = c(-1, 0.5, 1.5, 1, 0.2, 0.3, 0.4, 5)))
ctrl = makeTuneControlGrid()
rdesc = makeResampleDesc("Holdout")
learn = makeLearner("classif.ksvm", par.vals = list(kernel = "rbfdot"))
res = tuneParams(learn, task = pid.task, control = ctrl, measures = acc,
                 resampling = rdesc, par.set = ps, show.info = F)
data = generateHyperParsEffectData(res)
plotHyperParsEffect(data, x = "C", y = "sigma", z = "acc.test.mean",
                    plot.type = "heatmap", interpolate = TRUE)
```

![](tutorial_draft_files/figure-markdown_github/numericscrash-1.png)

A slightly more complicated example is using nested cross validation while simultaneously tuning 2 hyperparameters. In order to plot a heatmap in this case, mlr will aggregate each of the nested runs by a user-specified function. The default function is mean. As expected, we can still take advantage of interpolation.

``` r
ps = makeParamSet(
  makeNumericParam("C", lower = -5, upper = 5, trafo = function(x) 2^x),
  makeNumericParam("sigma", lower = -5, upper = 5, trafo = function(x) 2^x))
ctrl = makeTuneControlRandom(maxit = 100)
rdesc = makeResampleDesc("Holdout")
learn = makeLearner("classif.ksvm", par.vals = list(kernel = "rbfdot"))
lrn = makeTuneWrapper(learn, control = ctrl, 
           measures = list(acc, mmce), resampling = rdesc, par.set = ps, 
           show.info = F)
res = resample(lrn, task = pid.task, resampling = cv2, extract = getTuneResult)
data = generateHyperParsEffectData(res)
plotHyperParsEffect(data, x = "C", y = "sigma", z = "acc.test.mean",
                    plot.type = "heatmap", interpolate = TRUE, 
                    show.experiments = TRUE, nested.agg = mean)
```

![](tutorial_draft_files/figure-markdown_github/numericsnested-1.png)

Visualizing the effect of more than 2 hyperparameters
-----------------------------------------------------

The usecase here should reference partial dependency.

``` r
# convert to reg task
# call mlr partial dep
# provide feature selection?
```
