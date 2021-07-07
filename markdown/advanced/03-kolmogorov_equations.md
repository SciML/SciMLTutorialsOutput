---
author: "Ashutosh Bharambe"
title: "Kolmogorov Backward Equations"
---
```julia
using Flux, StochasticDiffEq
using NeuralPDE
using Plots
using CUDA
```



## Introduction on Backward Kolmogorov Equations

The backward Kolmogorov Equation deals with a terminal condtion.
The one dimensional backward kolmogorov equation that we are going to deal with is of the form :

$$
  \frac{\partial p}{\partial t} = -\mu(x)\frac{\partial p}{\partial x} - \frac{1}{2}{\sigma^2}(x)\frac{\partial^2 p}{\partial x^2} ,\hspace{0.5cm} p(T , x) = \varphi(x)
$$
for all $ t \in{ [0 , T] } $ and for all $ x \in R^d $

#### The Black Scholes Model

The Black-Scholes Model governs the price evolution of the European put or call option. In the below equation V is the price of some derivative , S is the Stock Price , r is the risk free interest
rate and σ the volatility of the stock returns. The payoff at a time T is known to us. And this makes it a terminal PDE. In case of an European put option the PDE is:
$$
  \frac{\partial V}{\partial t} + rS\frac{\partial V}{\partial S} + \frac{1}{2}{\sigma^2}{S^2}\frac{\partial^2 V}{\partial S^2} -rV = 0  ,\hspace{0.5cm} V(T , S) =  max\{\mathcal{K} - S , 0 \}
$$
for all $ t \in{ [0 , T] } $ and for all $ S \in R^d $

In order to make the above equation in the form of the Backward - Kolmogorov PDE we should substitute

$$
  V(S , t) = e^{r(t-T)}p(S , t)
$$
and thus we get
$$
  e^{r(t-T)}\frac{\partial p}{\partial t} + re^{r(t-T)}p(S , t)  = -\mu(x)\frac{\partial p}{\partial x}e^{r(t-T)} - \frac{1}{2}{\sigma^2}(x)\frac{\partial^2 p}{\partial x^2}e^{r(t-T)}
  + re^{r(t-T)}p(S , t)
$$
And the terminal condition
$$
  p(S , T) = max\{ \mathcal{K} - x , 0 \}
$$
We will train our model and the model itself will be the solution of the equation
## Defining the problem and the solver
We should start defining the terminal condition for our equation:
```julia
function phi(xi)
    y = Float64[]
    K = 100
    for x in eachcol(xi)
        val = max(K - maximum(x) , 0.00)
        y = push!(y , val)
    end
    y = reshape(y , 1 , size(y)[1] )
    return y
end
```

```
phi (generic function with 1 method)
```




Now we shall define the problem :
We will define the σ and μ by comparing it to the orignal equation. The xspan is the span of initial stock prices.
```julia
d = 1
r = 0.04
sigma = 0.2
xspan = (80.00 , 115.0)
tspan = (0.0 , 1.0)
σ(du , u , p , t) = du .= sigma.*u
μ(du , u , p , t) = du .= r.*u
prob = KolmogorovPDEProblem(μ , σ , phi , xspan , tspan, d)
```

```
KolmogorovPDEProblem with uType Int64 and tType Float64. In-place: nothing
timespan: (0.0, 1.0)
u0: 0
```




Now once we have defined our problem it is necessary to define the parameters for the solver.
```julia
sdealg = EM()
ensemblealg = EnsembleThreads()
dt = 0.01
dx = 0.01
trajectories = 100000
```

```
100000
```





Now lets define our model m and the optimiser
```julia
m = Chain(Dense(d, 64, elu),Dense(64, 128, elu),Dense(128 , 16 , elu) , Dense(16 , 1))
use_gpu = false
if CUDA.functional() == true
  m = fmap(CUDA.cu , m)
  use_gpu = true
end
opt = Flux.ADAM(0.0005)
```

```
Flux.Optimise.ADAM(0.0005, (0.9, 0.999), IdDict{Any, Any}())
```




And then finally call the solver
```julia
@time sol = solve(prob, NeuralPDE.NNKolmogorov(m, opt, sdealg, ensemblealg), verbose = true, dt = dt,
            dx = dx , trajectories = trajectories , abstol=1e-6, maxiters = 1000 , use_gpu = use_gpu)
```

```
Current loss is: 140.22230935704445
Current loss is: 136.45739161646566
Current loss is: 136.50506985652234
Current loss is: 137.74080966723628
Current loss is: 136.6720976793749
Current loss is: 135.36652603055873
Current loss is: 135.4041129876246
Current loss is: 135.9149969441637
Current loss is: 135.61463661180048
Current loss is: 134.7947012436686
Current loss is: 134.39022016501238
Current loss is: 134.6037459747041
Current loss is: 134.73543137467001
Current loss is: 134.31246819450874
Current loss is: 133.71634278394916
Current loss is: 133.47486407690175
Current loss is: 133.5268224494111
Current loss is: 133.41334609777442
Current loss is: 133.02850524932563
Current loss is: 132.75261681085718
Current loss is: 132.7926598406109
Current loss is: 132.63039807273734
Current loss is: 132.24752194116854
Current loss is: 132.00039549607865
Current loss is: 131.87738015475932
Current loss is: 131.67773656410932
Current loss is: 131.3806051010411
Current loss is: 131.11958716331563
Current loss is: 130.9087716459069
Current loss is: 130.56111078093343
Current loss is: 130.04944868809767
Current loss is: 129.67825467032702
Current loss is: 129.34426665755424
Current loss is: 129.12084983050863
Current loss is: 128.79473679250543
Current loss is: 128.54758055568192
Current loss is: 128.29498717709004
Current loss is: 127.9515068976029
Current loss is: 127.69565331164424
Current loss is: 127.42411634279918
Current loss is: 127.13716248552473
Current loss is: 126.86321712712589
Current loss is: 126.51141941308569
Current loss is: 126.16140368419562
Current loss is: 125.8564080174976
Current loss is: 125.57630656148531
Current loss is: 125.2965603375496
Current loss is: 124.98331316188153
Current loss is: 124.66707612597328
Current loss is: 124.3293874102636
Current loss is: 124.02982542818705
Current loss is: 123.6609720116771
Current loss is: 123.38353807243178
Current loss is: 123.10656653698895
Current loss is: 122.8565601349614
Current loss is: 122.56444369717781
Current loss is: 122.2813069840762
Current loss is: 121.96266491935637
Current loss is: 121.69385517674607
Current loss is: 121.42434844664574
Current loss is: 121.15951778578628
Current loss is: 120.90932300555865
Current loss is: 120.62672558437917
Current loss is: 120.35737283216615
Current loss is: 120.0734550526937
Current loss is: 119.77117594187595
Current loss is: 119.48634959297006
Current loss is: 119.19738281744446
Current loss is: 118.88641394998194
Current loss is: 118.59852903707453
Current loss is: 118.2959961452065
Current loss is: 117.96719142020766
Current loss is: 117.64314326546818
Current loss is: 117.31029560284442
Current loss is: 116.98620570679546
Current loss is: 116.70086414518721
Current loss is: 116.4256232778867
Current loss is: 116.14649333389514
Current loss is: 115.86202611302046
Current loss is: 115.56986818836843
Current loss is: 115.281380877951
Current loss is: 115.00290484698589
Current loss is: 114.73175690872121
Current loss is: 114.46208942621114
Current loss is: 114.18789288538896
Current loss is: 113.91284304157179
Current loss is: 113.63872792400247
Current loss is: 113.36847827868039
Current loss is: 113.10185053723241
Current loss is: 112.83854058084218
Current loss is: 112.57830824750654
Current loss is: 112.32178391779314
Current loss is: 112.07257683667825
Current loss is: 111.84188019275709
Current loss is: 111.67686591909272
Current loss is: 111.62419992899565
Current loss is: 111.5820605544207
Current loss is: 111.03600441332225
Current loss is: 110.67297133672699
Current loss is: 110.74315373978177
Current loss is: 110.39802457154981
Current loss is: 110.03700391801084
Current loss is: 110.05121614690673
Current loss is: 109.76121746131513
Current loss is: 109.45996149739162
Current loss is: 109.43577189434909
Current loss is: 109.17786205298624
Current loss is: 108.92058793091178
Current loss is: 108.86974903246428
Current loss is: 108.6509663679
Current loss is: 108.41555148490461
Current loss is: 108.34447919936657
Current loss is: 108.1744842548332
Current loss is: 107.9505194027133
Current loss is: 107.85442024183276
Current loss is: 107.7372196807613
Current loss is: 107.5364863250871
Current loss is: 107.40237382995238
Current loss is: 107.31926090940108
Current loss is: 107.17435249542478
Current loss is: 107.01238610848509
Current loss is: 106.9115455298055
Current loss is: 106.82680160832915
Current loss is: 106.70108571815318
Current loss is: 106.56565918010504
Current loss is: 106.4677309102884
Current loss is: 106.39323999955803
Current loss is: 106.3041541641349
Current loss is: 106.19569378766924
Current loss is: 106.0916901393791
Current loss is: 106.0095405082405
Current loss is: 105.9439176525788
Current loss is: 105.88013479082012
Current loss is: 105.80977254955724
Current loss is: 105.73269716333395
Current loss is: 105.65556188839048
Current loss is: 105.58380226161405
Current loss is: 105.52009359380823
Current loss is: 105.46403971515093
Current loss is: 105.41434540162435
Current loss is: 105.37032277276121
Current loss is: 105.33236682778775
Current loss is: 105.30282894955091
Current loss is: 105.28342112119415
Current loss is: 105.27923581324033
Current loss is: 105.2788733537841
Current loss is: 105.27325801078177
Current loss is: 105.21013598209687
Current loss is: 105.1035557252153
Current loss is: 104.99536098161742
Current loss is: 104.95418191794656
Current loss is: 104.9684460341752
Current loss is: 104.96431644617249
Current loss is: 104.90814449403383
Current loss is: 104.83871129872871
Current loss is: 104.81406202659785
Current loss is: 104.8183647597095
Current loss is: 104.79899074163097
Current loss is: 104.75073453381457
Current loss is: 104.70655780227119
Current loss is: 104.69141233764334
Current loss is: 104.69048692278149
Current loss is: 104.67551709497492
Current loss is: 104.64251049215461
Current loss is: 104.60708359238716
Current loss is: 104.58688774452709
Current loss is: 104.57938772799838
Current loss is: 104.56896645603605
Current loss is: 104.54762266997136
Current loss is: 104.5216845537458
Current loss is: 104.50302415273212
Current loss is: 104.49317065293387
Current loss is: 104.48402302928842
Current loss is: 104.4693831514991
Current loss is: 104.45020503295902
Current loss is: 104.43267279423601
Current loss is: 104.42017619515124
Current loss is: 104.41133059156161
Current loss is: 104.40243071092789
Current loss is: 104.39078253731981
Current loss is: 104.37687833199625
Current loss is: 104.36285404549342
Current loss is: 104.35091335867435
Current loss is: 104.34151520834705
Current loss is: 104.33345495095551
Current loss is: 104.32514036945688
Current loss is: 104.31566520265561
Current loss is: 104.30538222863413
Current loss is: 104.29507840535359
Current loss is: 104.2855566416779
Current loss is: 104.27709018524462
Current loss is: 104.26951461372339
Current loss is: 104.26245767065186
Current loss is: 104.25556413091395
Current loss is: 104.24862714798641
Current loss is: 104.24153009690914
Current loss is: 104.23432953555228
Current loss is: 104.22707125745806
Current loss is: 104.21990837781317
Current loss is: 104.21293017002546
Current loss is: 104.2062297120272
Current loss is: 104.19982872281366
Current loss is: 104.19371200757746
Current loss is: 104.18783725746576
Current loss is: 104.18216324493955
Current loss is: 104.17666694696744
Current loss is: 104.1713476113005
Current loss is: 104.16621584407075
Current loss is: 104.16129312166692
Current loss is: 104.15661474109139
Current loss is: 104.15224636920415
Current loss is: 104.14834549667668
Current loss is: 104.14515598392491
Current loss is: 104.143270522232
Current loss is: 104.14350527746625
Current loss is: 104.14805557082354
Current loss is: 104.15936290289761
Current loss is: 104.1850202452158
Current loss is: 104.22720786229063
Current loss is: 104.30130270960122
Current loss is: 104.36521977691496
Current loss is: 104.4102685914045
Current loss is: 104.31488208982188
Current loss is: 104.17716748657902
Current loss is: 104.09512337048797
Current loss is: 104.13550468973462
Current loss is: 104.22283935362368
Current loss is: 104.23104372826774
Current loss is: 104.16328644049868
Current loss is: 104.08626627050698
Current loss is: 104.08667666733146
Current loss is: 104.13897493737389
Current loss is: 104.15595651260276
Current loss is: 104.11772656712314
Current loss is: 104.06882017106835
Current loss is: 104.0692335958625
Current loss is: 104.10135761284137
Current loss is: 104.10782957305318
Current loss is: 104.08053900269857
Current loss is: 104.05232860246026
Current loss is: 104.05458354484708
Current loss is: 104.07354417592555
Current loss is: 104.07666122121692
Current loss is: 104.0594568291852
Current loss is: 104.04073572014893
Current loss is: 104.0396632548407
Current loss is: 104.0500340095075
Current loss is: 104.05284830073394
Current loss is: 104.04308528131754
Current loss is: 104.03050937226422
Current loss is: 104.02722071410123
Current loss is: 104.03208083046862
Current loss is: 104.03516641819935
Current loss is: 104.03110199132584
Current loss is: 104.02260126331718
Current loss is: 104.01706772409935
Current loss is: 104.01717010680373
Current loss is: 104.01947352513596
Current loss is: 104.01927453481473
Current loss is: 104.01507210555323
Current loss is: 104.00986554813288
Current loss is: 104.00680809374467
Current loss is: 104.00656036572639
Current loss is: 104.00720716239945
Current loss is: 104.00646354436479
Current loss is: 104.00385153840065
Current loss is: 104.00038226282193
Current loss is: 103.9976614300712
Current loss is: 103.99635596626065
Current loss is: 103.99597644500544
Current loss is: 103.99551262386065
Current loss is: 103.99422532790184
Current loss is: 103.99217967787467
Current loss is: 103.98987852327535
Current loss is: 103.98792821425248
Current loss is: 103.98657858112198
Current loss is: 103.98569700130781
Current loss is: 103.98494484357066
Current loss is: 103.98400132950837
Current loss is: 103.98274774029808
Current loss is: 103.9812249561686
Current loss is: 103.97961978754692
Current loss is: 103.97809946619726
Current loss is: 103.97677255874474
Current loss is: 103.97565105863724
Current loss is: 103.97467851865828
Current loss is: 103.97377219107759
Current loss is: 103.97285650872257
Current loss is: 103.97189141220579
Current loss is: 103.97085804586199
Current loss is: 103.96977407152022
Current loss is: 103.96865416325514
Current loss is: 103.9675284553112
Current loss is: 103.96641243272067
Current loss is: 103.96532321670276
Current loss is: 103.96426568225576
Current loss is: 103.96324233732551
Current loss is: 103.96225011576779
Current loss is: 103.96128515348715
Current loss is: 103.96034323732061
Current loss is: 103.9594211186408
Current loss is: 103.95851631840574
Current loss is: 103.95762681445171
Current loss is: 103.95675103806943
Current loss is: 103.95588790399736
Current loss is: 103.95503694820303
Current loss is: 103.95419809069448
Current loss is: 103.95337143119762
Current loss is: 103.95255731047207
Current loss is: 103.95175663864279
Current loss is: 103.9509715867104
Current loss is: 103.95020631953493
Current loss is: 103.94946896956111
Current loss is: 103.94877534317888
Current loss is: 103.94815590655371
Current loss is: 103.94767422190324
Current loss is: 103.94745339775082
Current loss is: 103.94776504168384
Current loss is: 103.94912613152209
Current loss is: 103.95277014678125
Current loss is: 103.96091578027003
Current loss is: 103.97936236276381
Current loss is: 104.01649018458633
Current loss is: 104.09713831717946
Current loss is: 104.22948231677215
Current loss is: 104.46145661993394
Current loss is: 104.57977887495467
Current loss is: 104.54053310531988
Current loss is: 104.143727703445
Current loss is: 103.94486367112626
Current loss is: 104.11254420029323
Current loss is: 104.29069065170812
Current loss is: 104.22205314272156
Current loss is: 103.98104351126022
Current loss is: 103.96716009526621
Current loss is: 104.13250801451198
Current loss is: 104.13681058691486
Current loss is: 103.99179717687275
Current loss is: 103.9412214744582
Current loss is: 104.04197802030929
Current loss is: 104.07226545383867
Current loss is: 103.96661773921947
Current loss is: 103.93902974833998
Current loss is: 104.01216568075651
Current loss is: 104.01889459721504
Current loss is: 103.95294188554166
Current loss is: 103.93354555122622
Current loss is: 103.97913269673357
Current loss is: 103.98908914051692
Current loss is: 103.94258035933703
Current loss is: 103.93123110862528
Current loss is: 103.96291185044419
Current loss is: 103.96432059306466
Current loss is: 103.93449820735603
Current loss is: 103.9278836104466
Current loss is: 103.94840156093348
Current loss is: 103.95189854669869
Current loss is: 103.93138917526277
Current loss is: 103.9245741168309
Current loss is: 103.93752642349575
Current loss is: 103.94091408472232
Current loss is: 103.92854826733938
Current loss is: 103.92223087321554
Current loss is: 103.92975691884034
Current loss is: 103.93393338364649
Current loss is: 103.92634395852876
Current loss is: 103.92038780632136
Current loss is: 103.92395491600159
Current loss is: 103.92795770420615
Current loss is: 103.9246165917309
Current loss is: 103.91929375383012
Current loss is: 103.91974627156226
Current loss is: 103.9230660526675
Current loss is: 103.92236974776819
Current loss is: 103.91856919862602
Current loss is: 103.9171481099613
Current loss is: 103.91896714475602
Current loss is: 103.91987123422723
Current loss is: 103.91786578327456
Current loss is: 103.91573434044496
Current loss is: 103.91587232088031
Current loss is: 103.91695608839989
Current loss is: 103.9166075922108
Current loss is: 103.91494536408226
Current loss is: 103.91396972536238
Current loss is: 103.91432332023862
Current loss is: 103.91470232010376
Current loss is: 103.91405869340527
Current loss is: 103.91292352167248
Current loss is: 103.91240025145267
Current loss is: 103.91257375123251
Current loss is: 103.91261823257211
Current loss is: 103.91205437205281
Current loss is: 103.91126784797281
Current loss is: 103.91086262646452
Current loss is: 103.91085431550378
Current loss is: 103.91077682692122
Current loss is: 103.91035212512638
Current loss is: 103.90977146972824
Current loss is: 103.90937910128068
Current loss is: 103.9092361684865
Current loss is: 103.9091131750454
Current loss is: 103.90881078212651
Current loss is: 103.90837010992233
Current loss is: 103.90798072647644
Current loss is: 103.90774145790482
Current loss is: 103.9075764197696
Current loss is: 103.90735194666833
Current loss is: 103.90702497688349
Current loss is: 103.90666829852637
Current loss is: 103.90637127748887
Current loss is: 103.90615017123655
Current loss is: 103.90594812197298
Current loss is: 103.90570444489596
Current loss is: 103.90541206341155
Current loss is: 103.9051113163824
Current loss is: 103.90484415803115
Current loss is: 103.90461659905705
Current loss is: 103.90440166157146
Current loss is: 103.90416943106483
Current loss is: 103.9039120049531
Current loss is: 103.90364533083637
Current loss is: 103.90339082322515
Current loss is: 103.90315772806782
Current loss is: 103.90293875854282
Current loss is: 103.90271931816585
Current loss is: 103.90248976877298
Current loss is: 103.90225092573168
Current loss is: 103.90201110828067
Current loss is: 103.90177863531905
Current loss is: 103.90155649521377
Current loss is: 103.90134172095796
Current loss is: 103.901128629955
Current loss is: 103.90091286323134
Current loss is: 103.90069349345924
Current loss is: 103.90047278200542
Current loss is: 103.9002541039373
Current loss is: 103.90003992060684
Current loss is: 103.89983070679962
Current loss is: 103.89962519226975
Current loss is: 103.89942140374184
Current loss is: 103.89921776811387
Current loss is: 103.8990136933915
Current loss is: 103.89880956061069
Current loss is: 103.89860627643193
Current loss is: 103.89840475929805
Current loss is: 103.89820559085898
Current loss is: 103.89800893091564
Current loss is: 103.89781453818536
Current loss is: 103.89762199027307
Current loss is: 103.8974308212965
Current loss is: 103.8972407022189
Current loss is: 103.89705147901331
Current loss is: 103.89686315406047
Current loss is: 103.89667583870803
Current loss is: 103.89648969117515
Current loss is: 103.89630482140853
Current loss is: 103.89612130817879
Current loss is: 103.89593920680389
Current loss is: 103.89575853557464
Current loss is: 103.89557926199416
Current loss is: 103.89540135300169
Current loss is: 103.89522476058634
Current loss is: 103.89504945027088
Current loss is: 103.89487540175111
Current loss is: 103.89470257823207
Current loss is: 103.8945309629805
Current loss is: 103.89436055440116
Current loss is: 103.89419136570746
Current loss is: 103.89402339736498
Current loss is: 103.89385667714596
Current loss is: 103.89369126948442
Current loss is: 103.8935272783753
Current loss is: 103.89336483038075
Current loss is: 103.89320418444535
Current loss is: 103.89304574842099
Current loss is: 103.89289018575256
Current loss is: 103.89273861619189
Current loss is: 103.89259295205139
Current loss is: 103.89245642940901
Current loss is: 103.89233479843139
Current loss is: 103.8922376985658
Current loss is: 103.89218334058627
Current loss is: 103.89220240997996
Current loss is: 103.8923557754986
Current loss is: 103.89274510707905
Current loss is: 103.89358552768242
Current loss is: 103.89522463772484
Current loss is: 103.89846476730392
Current loss is: 103.90449690720354
Current loss is: 103.91644069045043
Current loss is: 103.93801945613414
Current loss is: 103.98116082384833
Current loss is: 104.05198260900092
Current loss is: 104.1866860677357
Current loss is: 104.33937895739152
Current loss is: 104.54519332264807
Current loss is: 104.49364900272654
Current loss is: 104.29272236718778
Current loss is: 103.97469313886795
Current loss is: 103.90757892194901
Current loss is: 104.0884132431097
Current loss is: 104.24083136389427
Current loss is: 104.22639663058914
Current loss is: 104.00898587693513
Current loss is: 103.88940319253162
Current loss is: 103.96606948379997
Current loss is: 104.07130257760731
Current loss is: 104.04660588522748
Current loss is: 103.92198707030069
Current loss is: 103.89795222077515
Current loss is: 103.97847172547753
Current loss is: 104.01158675019019
Current loss is: 103.9565508641617
Current loss is: 103.89230041029349
Current loss is: 103.9080391392084
Current loss is: 103.95846277913876
Current loss is: 103.95196682321036
Current loss is: 103.90567672227563
Current loss is: 103.8892585665853
Current loss is: 103.91782464427844
Current loss is: 103.93692602809404
Current loss is: 103.91440282479775
Current loss is: 103.88900345942464
Current loss is: 103.89375571193926
Current loss is: 103.91303565497222
Current loss is: 103.91456066713855
Current loss is: 103.89605197717123
Current loss is: 103.8864157465221
Current loss is: 103.89557811298859
Current loss is: 103.90477734342113
Current loss is: 103.89977854777914
Current loss is: 103.88837986600042
Current loss is: 103.88649842437033
Current loss is: 103.89367938843506
Current loss is: 103.89730675749055
Current loss is: 103.89242194446959
Current loss is: 103.88598925289578
Current loss is: 103.88612773078427
Current loss is: 103.89064050281151
Current loss is: 103.89198519367986
Current loss is: 103.88850554111782
Current loss is: 103.88490335458465
Current loss is: 103.88527713286392
Current loss is: 103.88793659684248
Current loss is: 103.88869253151644
Current loss is: 103.8866303036848
Current loss is: 103.88434306163822
Current loss is: 103.88429439393792
Current loss is: 103.88578624496444
Current loss is: 103.88643880460099
Current loss is: 103.88539895802829
Current loss is: 103.88390143563895
Current loss is: 103.88350387765948
Current loss is: 103.88421164769814
Current loss is: 103.88481351578824
Current loss is: 103.88448442245812
Current loss is: 103.88355409274125
Current loss is: 103.88296366516185
Current loss is: 103.88310206227885
Current loss is: 103.88352401558537
Current loss is: 103.88359336386884
Current loss is: 103.88315784077534
Current loss is: 103.88261947996855
Current loss is: 103.88239623298425
Current loss is: 103.88252181646627
Current loss is: 103.88269144734893
Current loss is: 103.88261932324943
Current loss is: 103.88231198857501
Current loss is: 103.88199080663212
Current loss is: 103.88185124507345
Current loss is: 103.88188650430054
Current loss is: 103.8819377074176
Current loss is: 103.88186748232886
Current loss is: 103.88167024202414
Current loss is: 103.88145251508969
Current loss is: 103.88131599343158
Current loss is: 103.88127668114413
Current loss is: 103.88127002429196
Current loss is: 103.88121987775196
Current loss is: 103.88110175181671
Current loss is: 103.8809480475536
Current loss is: 103.88081227854266
Current loss is: 103.88072377381619
Current loss is: 103.88067262078043
Current loss is: 103.88062554249521
Current loss is: 103.88055375109721
Current loss is: 103.88045193654317
Current loss is: 103.88033596363346
Current loss is: 103.88022840857884
Current loss is: 103.88014212194443
Current loss is: 103.88007431715793
Current loss is: 103.88001181258153
Current loss is: 103.87994137827526
Current loss is: 103.8798577704621
Current loss is: 103.87976459079604
Current loss is: 103.87967029484719
Current loss is: 103.87958224165568
Current loss is: 103.87950318856755
Current loss is: 103.87943098559047
Current loss is: 103.87936080674712
Current loss is: 103.87928808851402
Current loss is: 103.87921061385151
Current loss is: 103.87912905378467
Current loss is: 103.87904592118241
Current loss is: 103.87896401611373
Current loss is: 103.8788851124762
Current loss is: 103.87880952009667
Current loss is: 103.87873636879188
Current loss is: 103.87866424750183
Current loss is: 103.87859184159096
Current loss is: 103.87851833902914
Current loss is: 103.87844357946693
Current loss is: 103.87836796077016
Current loss is: 103.87829217599202
Current loss is: 103.8782168562781
Current loss is: 103.87814243147109
Current loss is: 103.87806906992265
Current loss is: 103.87799674164107
Current loss is: 103.87792525593154
Current loss is: 103.87785435555777
Current loss is: 103.87778378664214
Current loss is: 103.87771335623972
Current loss is: 103.87764296794894
Current loss is: 103.87757260067563
Current loss is: 103.8775022783254
Current loss is: 103.87743204312493
Current loss is: 103.87736194959771
Current loss is: 103.8772920754839
Current loss is: 103.87722247071963
Current loss is: 103.87715317068388
Current loss is: 103.87708417874731
Current loss is: 103.87701551610478
Current loss is: 103.87694718701407
Current loss is: 103.87687918289427
Current loss is: 103.87681150033858
Current loss is: 103.876744131736
Current loss is: 103.87667707598902
Current loss is: 103.87661033142071
Current loss is: 103.87654389662184
Current loss is: 103.87647779314807
Current loss is: 103.87641203094938
Current loss is: 103.87634664385573
Current loss is: 103.87628168310484
Current loss is: 103.87621722440554
Current loss is: 103.87615338088446
Current loss is: 103.87609034873164
Current loss is: 103.87602844332982
Current loss is: 103.87596818067344
Current loss is: 103.87591045545838
Current loss is: 103.87585668078884
Current loss is: 103.87580931840623
Current loss is: 103.8757724480874
Current loss is: 103.8757534387239
Current loss is: 103.87576443226028
Current loss is: 103.87582850376431
Current loss is: 103.8759837136961
Current loss is: 103.87630542299382
Current loss is: 103.87691611690938
Current loss is: 103.87807605814984
Current loss is: 103.88018654928163
Current loss is: 103.88418772906337
Current loss is: 103.89137595364171
Current loss is: 103.9052647061031
Current loss is: 103.92954585138686
Current loss is: 103.97715508785645
Current loss is: 104.05257986453856
Current loss is: 104.19516164581268
Current loss is: 104.35327679605771
Current loss is: 104.5801855303486
Current loss is: 104.55135782432335
Current loss is: 104.38477621640266
Current loss is: 104.02392993412084
Current loss is: 103.879094770046
Current loss is: 104.01789489404328
Current loss is: 104.22456077310436
Current loss is: 104.31979557503006
Current loss is: 104.12753038985021
Current loss is: 103.92109207265052
Current loss is: 103.89150393181961
Current loss is: 104.01968636253899
Current loss is: 104.09364761104428
Current loss is: 103.98556144152792
Current loss is: 103.88120058773298
Current loss is: 103.90521594055403
Current loss is: 103.99000919081028
Current loss is: 104.01059450842988
Current loss is: 103.93082846891899
Current loss is: 103.87509228404977
Current loss is: 103.90393983561933
Current loss is: 103.94960881925152
Current loss is: 103.93834391474184
Current loss is: 103.8878162406531
Current loss is: 103.87687362614056
Current loss is: 103.90900258078253
Current loss is: 103.9256774910374
Current loss is: 103.9050317680265
Current loss is: 103.87680539001619
Current loss is: 103.87923317395061
Current loss is: 103.89991665376719
Current loss is: 103.90235150145547
Current loss is: 103.88497438214081
Current loss is: 103.87334410600491
Current loss is: 103.88117941363112
Current loss is: 103.89262736373864
Current loss is: 103.8895611726856
Current loss is: 103.87789499065212
Current loss is: 103.8730822847447
Current loss is: 103.87910393053272
Current loss is: 103.88488268606544
Current loss is: 103.88153010833356
Current loss is: 103.8746016197157
Current loss is: 103.87302268776227
Current loss is: 103.87719018500782
Current loss is: 103.8801484135845
Current loss is: 103.87771115830138
Current loss is: 103.87353275287101
Current loss is: 103.87256807851496
Current loss is: 103.87496452123281
Current loss is: 103.87675189033713
Current loss is: 103.87542811897993
Current loss is: 103.87292718330238
Current loss is: 103.87216487252844
Current loss is: 103.8734587329102
Current loss is: 103.87467258621477
Current loss is: 103.87416648718252
Current loss is: 103.8726645210179
Current loss is: 103.87184771135476
Current loss is: 103.87233088665677
Current loss is: 103.87315799101486
Current loss is: 103.87317840456326
Current loss is: 103.87239738488991
Current loss is: 103.87167957579246
Current loss is: 103.87166595238179
Current loss is: 103.87212290719657
Current loss is: 103.87237548440476
Current loss is: 103.87211045821888
Current loss is: 103.87159895966873
Current loss is: 103.87132087710647
Current loss is: 103.87142555313454
Current loss is: 103.87165688310009
Current loss is: 103.87169284498891
Current loss is: 103.87146439919324
Current loss is: 103.87117546647178
Current loss is: 103.87104431827912
Current loss is: 103.8711028893022
Current loss is: 103.87120457413711
Current loss is: 103.87119275307651
Current loss is: 103.87104706692418
Current loss is: 103.8708702292995
Current loss is: 103.87077511890476
Current loss is: 103.87078160938272
Current loss is: 103.87081919422899
Current loss is: 103.87080647947643
Current loss is: 103.87072080025655
Current loss is: 103.87060578212403
Current loss is: 103.87051983185154
Current loss is: 103.87048681382595
Current loss is: 103.87048464657616
Current loss is: 103.87047268171936
Current loss is: 103.8704270286975
Current loss is: 103.87035396201294
Current loss is: 103.87027905641146
Current loss is: 103.87022407092682
Current loss is: 103.87019271028144
Current loss is: 103.87017179975689
Current loss is: 103.87014369125558
Current loss is: 103.87009946959343
Current loss is: 103.87004304218247
Current loss is: 103.8699860098149
Current loss is: 103.86993804792844
Current loss is: 103.86990115945478
Current loss is: 103.86987017475023
Current loss is: 103.8698376622272
Current loss is: 103.86979892140738
Current loss is: 103.86975393141094
Current loss is: 103.8697063267297
Current loss is: 103.86966050157925
Current loss is: 103.86961912387957
Current loss is: 103.86958208203666
Current loss is: 103.86954705762977
Current loss is: 103.86951123523377
Current loss is: 103.86947290287029
Current loss is: 103.8694320671757
Current loss is: 103.8693899978194
Current loss is: 103.86934827718083
Current loss is: 103.86930804703772
Current loss is: 103.86926963468679
Current loss is: 103.86923258717702
Current loss is: 103.86919603786997
Current loss is: 103.86915911632568
Current loss is: 103.86912136071155
Current loss is: 103.86908279097496
Current loss is: 103.86904376142425
Current loss is: 103.86900474459598
Current loss is: 103.86896612637527
Current loss is: 103.86892810812768
Current loss is: 103.86889070903248
Current loss is: 103.8688537891405
Current loss is: 103.86881711074162
Current loss is: 103.86878046135094
Current loss is: 103.86874370760722
Current loss is: 103.86870680486031
Current loss is: 103.86866977028471
Current loss is: 103.8686326627456
Current loss is: 103.86859556622657
Current loss is: 103.8685585755999
Current loss is: 103.86852175125499
Current loss is: 103.86848512681077
Current loss is: 103.86844868984026
Current loss is: 103.8684124210148
Current loss is: 103.86837630106214
Current loss is: 103.86834029583517
Current loss is: 103.86830438955191
Current loss is: 103.86826854703799
Current loss is: 103.86823276564766
Current loss is: 103.86819703121593
Current loss is: 103.8681613451127
Current loss is: 103.86812569802268
Current loss is: 103.86809009554662
Current loss is: 103.8680545435654
Current loss is: 103.86801904212713
Current loss is: 103.8679835968503
Current loss is: 103.86794821357246
Current loss is: 103.86791290027166
Current loss is: 103.86787765922881
Current loss is: 103.8678424907958
Current loss is: 103.86780740316811
Current loss is: 103.86777239813604
Current loss is: 103.86773750051886
Current loss is: 103.86770273289002
Current loss is: 103.86766814006155
Current loss is: 103.86763378243633
Current loss is: 103.86759975606728
Current loss is: 103.86756620873886
Current loss is: 103.86753339251902
Current loss is: 103.86750169365756
Current loss is: 103.86747177196209
Current loss is: 103.86744464819763
Current loss is: 103.8674219192205
Current loss is: 103.86740622584328
Current loss is: 103.86740162203833
Current loss is: 103.86741531482818
Current loss is: 103.86745857072692
Current loss is: 103.86755214235052
Current loss is: 103.86772732655056
Current loss is: 103.86804576921924
Current loss is: 103.86859804029811
Current loss is: 103.86957616849632
Current loss is: 103.87124438515298
Current loss is: 103.87422930603182
Current loss is: 103.87929260195423
Current loss is: 103.8885727529556
Current loss is: 103.90402220354605
Current loss is: 103.93299459702317
Current loss is: 103.9782239046006
Current loss is: 104.06293328057829
Current loss is: 104.1710950480561
Current loss is: 104.354358628254
Current loss is: 104.45989040582225
Current loss is: 104.5507720691802
Current loss is: 104.33060897781529
Current loss is: 104.06423658626227
Current loss is: 103.87882203237918
Current loss is: 103.9229582753832
Current loss is: 104.1119726626579
Current loss is: 104.23938931738999
Current loss is: 104.253939686028
Current loss is: 104.05782326335739
Current loss is: 103.89404812732838
Current loss is: 103.89678323275315
Current loss is: 104.01095526221553
Current loss is: 104.07331902456708
Current loss is: 103.98920254730398
Current loss is: 103.89044663169045
Current loss is: 103.87369233649567
Current loss is: 103.93420100483921
Current loss is: 103.98479844736984
Current loss is: 103.95077103329649
Current loss is: 103.88806419077298
Current loss is: 103.87026235323209
Current loss is: 103.90680245853811
Current loss is: 103.93740969460163
Current loss is: 103.91708659814032
Current loss is: 103.87976832789164
Current loss is: 103.86729489712803
Current loss is: 103.8867435676525
Current loss is: 103.9062714030966
Current loss is: 103.89732183607362
Current loss is: 103.87507722048436
Current loss is: 103.86668186840531
Current loss is: 103.87833709516603
Current loss is: 103.89054556048427
Current loss is: 103.88623000228647
Current loss is: 103.87292496968175
Current loss is: 103.86586356369696
Current loss is: 103.87095637734048
Current loss is: 103.87891401522772
Current loss is: 103.87845325387093
Current loss is: 103.87110376806449
Current loss is: 103.865767587645
Current loss is: 103.86770623703427
Current loss is: 103.87275378931909
Current loss is: 103.87397354627308
Current loss is: 103.870393972812
Current loss is: 103.86618251106538
Current loss is: 103.86566178297356
Current loss is: 103.86826667105447
Current loss is: 103.87013446843554
Current loss is: 103.86911321477912
Current loss is: 103.86650002254628
Current loss is: 103.86516014443256
Current loss is: 103.86599112883745
Current loss is: 103.86749902987928
Current loss is: 103.86786460911375
Current loss is: 103.86672626956478
Current loss is: 103.8653652285702
Current loss is: 103.86499647861214
Current loss is: 103.86564924690694
Current loss is: 103.86637112957196
Current loss is: 103.86633230866292
Current loss is: 103.86562517159909
Current loss is: 103.86492299271698
Current loss is: 103.86478234449007
Current loss is: 103.86514257893361
Current loss is: 103.86550512961149
Current loss is: 103.86547446947556
Current loss is: 103.86507546057145
Current loss is: 103.86467201831015
Current loss is: 103.86455803149543
Current loss is: 103.86472239181712
Current loss is: 103.8649213678827
Current loss is: 103.8649322180327
Current loss is: 103.86474087471788
Current loss is: 103.86449728549051
Current loss is: 103.86436628869893
Current loss is: 103.86438930463478
Current loss is: 103.86448148357044
Current loss is: 103.86452553759901
Current loss is: 103.86446437353796
Current loss is: 103.86433295828799
Current loss is: 103.86420972777553
Current loss is: 103.86415405008974
Current loss is: 103.8641659038699
Current loss is: 103.86419749226081
Current loss is: 103.86419708758517
Current loss is: 103.86414582849534
Current loss is: 103.86406537956793
Current loss is: 103.86399206244958
Current loss is: 103.86394999350172
Current loss is: 103.86393823853145
Current loss is: 103.86393781137504
Current loss is: 103.86392742706357
Current loss is: 103.86389606267964
Current loss is: 103.86384759019751
Current loss is: 103.86379529075529
Current loss is: 103.86375261440973
Current loss is: 103.86372491142858
Current loss is: 103.86370794237155
Current loss is: 103.86369250343778
Current loss is: 103.86367081246041
Current loss is: 103.86364037803074
Current loss is: 103.86360369389884
Current loss is: 103.8635659016153
Current loss is: 103.863531610059
Current loss is: 103.86350300397844
Current loss is: 103.86347927376343
Current loss is: 103.86345748533131
Current loss is: 103.86343444020638
Current loss is: 103.86340829821519
Current loss is: 103.8633791317537
Current loss is: 103.86334825195678
Current loss is: 103.86331735910451
Current loss is: 103.86328782858043
Current loss is: 103.86326031738088
Current loss is: 103.8632346438985
Current loss is: 103.86320998717102
Current loss is: 103.86318533176723
Current loss is: 103.86315995394554
Current loss is: 103.86313360891775
Current loss is: 103.86310642851271
Current loss is: 103.86307877988486
Current loss is: 103.86305111219208
Current loss is: 103.863023809973
Current loss is: 103.86299710181574
Current loss is: 103.86297101584047
Current loss is: 103.86294540467371
Current loss is: 103.86292006448589
Current loss is: 103.86289481270704
Current loss is: 103.86286949260965
Current loss is: 103.86284400538031
Current loss is: 103.86281832527764
Current loss is: 103.86279249437555
Current loss is: 103.86276659665171
Current loss is: 103.86274070673497
Current loss is: 103.86271487431098
Current loss is: 103.86268915078495
Current loss is: 103.86266356587817
Current loss is: 103.86263812958893
Current loss is: 103.86261283115014
2364.779160 seconds (86.35 M allocations: 1.181 TiB, 4.15% gc time, 0.06% c
ompilation time)
([101.96 90.79 … 107.83 98.84], [5.576494112649722 10.838511599477496 … 3.7
991463828242145 6.696448719935372])
```




## Analyzing the solution
Now let us find a Monte-Carlo Solution and plot the both:
```julia
monte_carlo_sol = []
x_out = collect(85:2.00:110.00)
for x in x_out
  u₀= [x]
  g_val(du , u , p , t) = du .= 0.2.*u
  f_val(du , u , p , t) = du .= 0.04.*u
  dt = 0.01
  tspan = (0.0,1.0)
  prob = SDEProblem(f_val,g_val,u₀,tspan)
  output_func(sol,i) = (sol[end],false)
  ensembleprob_val = EnsembleProblem(prob , output_func = output_func )
  sim_val = solve(ensembleprob_val, EM(), EnsembleThreads() , dt=0.01, trajectories=100000,adaptive=false)
  s = reduce(hcat , sim_val.u)
  mean_phi = sum(phi(s))/length(phi(s))
  global monte_carlo_sol = push!(monte_carlo_sol , mean_phi)
end
```




##Plotting the Solutions
We should reshape the inputs and outputs to make it compatible with our model. This is the most important part. The algorithm gives a distributed function over all initial prices in the xspan.
```julia
x_model = reshape(x_out, 1 , size(x_out)[1])
if use_gpu == true
  m = fmap(cpu , m)
end
y_out = m(x_model)
y_out = reshape(y_out , 13 , 1)
```

```
13×1 Matrix{Float64}:
 14.667706042811282
 13.30161067022798
 11.982492411169394
 10.70646560906542
  9.50331545741199
  8.416853582076294
  7.458639782514835
  6.634381379590666
  5.903066067072977
  5.236962486928021
  4.617841280421979
  4.033501063857963
  3.475620143179423
```




And now finally we can plot the solutions
```julia
plot(x_out , y_out , lw = 3 ,  xaxis="Initial Stock Price", yaxis="Payoff" , label = "NNKolmogorov")
plot!(x_out , monte_carlo_sol , lw = 3 ,  xaxis="Initial Stock Price", yaxis="Payoff" ,label = "Monte Carlo Solutions")
```

![](figures/03-kolmogorov_equations_9_1.png)


## Appendix

These tutorials are a part of the SciMLTutorials.jl repository, found at: [https://github.com/SciML/SciMLTutorials.jl](https://github.com/SciML/SciMLTutorials.jl). For more information on high-performance scientific machine learning, check out the SciML Open Source Software Organization [https://sciml.ai](https://sciml.ai).

To locally run this tutorial, do the following commands:

```
using SciMLTutorials
SciMLTutorials.weave_file("tutorials/advanced","03-kolmogorov_equations.jmd")
```

Computer Information:

```
Julia Version 1.6.1
Commit 6aaedecc44 (2021-04-23 05:59 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: AMD EPYC 7502 32-Core Processor
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, znver2)
Environment:
  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/a6029d3a-f78b-41ea-bc97-28aa57c6c6ea
  JULIA_NUM_THREADS = 16

```

Package Information:

```
      Status `/var/lib/buildkite-agent/builds/1-amdci4-julia-csail-mit-edu/julialang/scimltutorials-dot-jl/tutorials/advanced/Project.toml`
  [2169fc97] AlgebraicMultigrid v0.4.0
  [6e4b80f9] BenchmarkTools v1.0.0
  [052768ef] CUDA v2.6.3
  [2b5f629d] DiffEqBase v6.62.2
  [9fdde737] DiffEqOperators v4.26.0
  [0c46a032] DifferentialEquations v6.17.1
  [587475ba] Flux v0.12.1
  [961ee093] ModelingToolkit v5.17.3
  [2774e3e8] NLsolve v4.5.1
  [315f7962] NeuralPDE v3.10.1
  [1dea7af3] OrdinaryDiffEq v5.56.0
  [91a5bcdd] Plots v1.15.2
  [0bca4576] SciMLBase v1.13.4
  [30cb0354] SciMLTutorials v0.9.0
  [47a9eef4] SparseDiffTools v1.13.2
  [684fba80] SparsityDetection v0.3.4
  [789caeaf] StochasticDiffEq v6.34.1
  [c3572dad] Sundials v4.4.3
  [37e2e46d] LinearAlgebra
  [2f01184e] SparseArrays
```

And the full manifest:

```
      Status `/var/lib/buildkite-agent/builds/1-amdci4-julia-csail-mit-edu/julialang/scimltutorials-dot-jl/tutorials/advanced/Manifest.toml`
  [c3fe647b] AbstractAlgebra v0.16.0
  [621f4979] AbstractFFTs v1.0.1
  [1520ce14] AbstractTrees v0.3.4
  [79e6a3ab] Adapt v3.3.0
  [2169fc97] AlgebraicMultigrid v0.4.0
  [ec485272] ArnoldiMethod v0.1.0
  [4fba245c] ArrayInterface v3.1.15
  [4c555306] ArrayLayouts v0.7.0
  [13072b0f] AxisAlgorithms v1.0.0
  [ab4f0b2a] BFloat16s v0.1.0
  [aae01518] BandedMatrices v0.16.9
  [6e4b80f9] BenchmarkTools v1.0.0
  [8e7c35d0] BlockArrays v0.15.3
  [ffab5731] BlockBandedMatrices v0.10.6
  [764a87c0] BoundaryValueDiffEq v2.7.1
  [fa961155] CEnum v0.4.1
  [00ebfdb7] CSTParser v2.5.0
  [052768ef] CUDA v2.6.3
  [7057c7e9] Cassette v0.3.6
  [082447d4] ChainRules v0.7.65
  [d360d2e6] ChainRulesCore v0.9.44
  [b630d9fa] CheapThreads v0.2.5
  [944b1d66] CodecZlib v0.7.0
  [35d6a980] ColorSchemes v3.12.1
  [3da002f7] ColorTypes v0.11.0
  [5ae59095] Colors v0.12.8
  [861a8166] Combinatorics v1.0.2
  [a80b9123] CommonMark v0.8.1
  [38540f10] CommonSolve v0.2.0
  [bbf7d656] CommonSubexpressions v0.3.0
  [34da2185] Compat v3.30.0
  [aa819f21] CompatHelper v1.18.6
  [8f4d0f93] Conda v1.5.2
  [88cd18e8] ConsoleProgressMonitor v0.1.2
  [187b0558] ConstructionBase v1.2.1
  [d38c429a] Contour v0.5.7
  [a8cc5b0e] Crayons v4.0.4
  [8a292aeb] Cuba v2.2.0
  [667455a9] Cubature v1.5.1
  [9a962f9c] DataAPI v1.6.0
  [82cc6244] DataInterpolations v3.3.1
  [864edb3b] DataStructures v0.18.9
  [e2d170a0] DataValueInterfaces v1.0.0
  [bcd4f6db] DelayDiffEq v5.31.0
  [2b5f629d] DiffEqBase v6.62.2
  [459566f4] DiffEqCallbacks v2.16.1
  [5a0ffddc] DiffEqFinancial v2.4.0
  [aae7a2af] DiffEqFlux v1.37.0
  [c894b116] DiffEqJump v6.14.2
  [77a26b50] DiffEqNoiseProcess v5.7.3
  [9fdde737] DiffEqOperators v4.26.0
  [055956cb] DiffEqPhysics v3.9.0
  [41bf760c] DiffEqSensitivity v6.45.0
  [163ba53b] DiffResults v1.0.3
  [b552c78f] DiffRules v1.0.2
  [0c46a032] DifferentialEquations v6.17.1
  [c619ae07] DimensionalPlotRecipes v1.2.0
  [b4f34e82] Distances v0.10.3
  [31c24e10] Distributions v0.24.18
  [ced4e74d] DistributionsAD v0.6.26
  [ffbed154] DocStringExtensions v0.8.4
  [e30172f5] Documenter v0.26.3
  [d4d017d3] ExponentialUtilities v1.8.4
  [e2ba6199] ExprTools v0.1.3
  [8f5d6c58] EzXML v1.1.0
  [c87230d0] FFMPEG v0.4.0
  [7a1cc6ca] FFTW v1.4.1
  [7034ab61] FastBroadcast v0.1.8
  [9aa1b823] FastClosures v0.3.2
  [1a297f60] FillArrays v0.11.7
  [6a86dc24] FiniteDiff v2.8.0
  [53c48c17] FixedPointNumbers v0.8.4
  [587475ba] Flux v0.12.1
  [59287772] Formatting v0.4.2
  [f6369f11] ForwardDiff v0.10.18
  [069b7b12] FunctionWrappers v1.1.2
  [d9f16b24] Functors v0.2.1
  [0c68f7d7] GPUArrays v6.4.1
  [61eb1bfa] GPUCompiler v0.10.0
  [28b8d3ca] GR v0.57.4
  [a75be94c] GalacticOptim v1.2.0
  [5c1252a2] GeometryBasics v0.3.12
  [bc5e4493] GitHub v5.4.0
  [af5da776] GlobalSensitivity v1.0.0
  [42e2da0e] Grisu v1.0.2
  [19dc6840] HCubature v1.5.0
  [cd3eb016] HTTP v0.9.9
  [eafb193a] Highlights v0.4.5
  [0e44f5e4] Hwloc v2.0.0
  [7073ff75] IJulia v1.23.2
  [b5f81e59] IOCapture v0.1.1
  [7869d1d1] IRTools v0.4.2
  [615f187c] IfElse v0.1.0
  [d25df0c9] Inflate v0.1.2
  [83e8ac13] IniFile v0.5.0
  [a98d9a8b] Interpolations v0.13.2
  [c8e1da08] IterTools v1.3.0
  [42fd0dbc] IterativeSolvers v0.9.1
  [82899510] IteratorInterfaceExtensions v1.0.0
  [692b3bcd] JLLWrappers v1.3.0
  [682c06a0] JSON v0.21.1
  [98e50ef6] JuliaFormatter v0.13.7
  [e5e0dc1b] Juno v0.8.4
  [5ab0869b] KernelDensity v0.6.3
  [929cbde3] LLVM v3.7.1
  [b964fa9f] LaTeXStrings v1.2.1
  [2ee39098] LabelledArrays v1.6.1
  [23fbe1c1] Latexify v0.15.5
  [a5e1c1ea] LatinHypercubeSampling v1.8.0
  [73f95e8e] LatticeRules v0.0.1
  [5078a376] LazyArrays v0.21.4
  [d7e5e226] LazyBandedMatrices v0.5.7
  [1d6d02ad] LeftChildRightSiblingTrees v0.1.2
  [093fc24a] LightGraphs v1.3.5
  [d3d80556] LineSearches v7.1.1
  [2ab3a3ac] LogExpFunctions v0.2.4
  [e6f89c97] LoggingExtras v0.4.6
  [bdcacae8] LoopVectorization v0.12.23
  [1914dd2f] MacroTools v0.5.6
  [a3b82374] MatrixFactorizations v0.8.3
  [739be429] MbedTLS v1.0.3
  [442fdcdd] Measures v0.3.1
  [e89f7d12] Media v0.5.0
  [c03570c3] Memoize v0.4.4
  [e1d29d7a] Missings v1.0.0
  [78c3b35d] Mocking v0.7.1
  [961ee093] ModelingToolkit v5.17.3
  [4886b29c] MonteCarloIntegration v0.0.2
  [46d2c3a1] MuladdMacro v0.2.2
  [f9640e96] MultiScaleArrays v1.8.1
  [ffc61752] Mustache v1.0.10
  [d41bc354] NLSolversBase v7.8.0
  [2774e3e8] NLsolve v4.5.1
  [872c559c] NNlib v0.7.19
  [77ba4419] NaNMath v0.3.5
  [315f7962] NeuralPDE v3.10.1
  [8913a72c] NonlinearSolve v0.3.8
  [6fe1bfb0] OffsetArrays v1.9.0
  [429524aa] Optim v1.3.0
  [bac558e1] OrderedCollections v1.4.1
  [1dea7af3] OrdinaryDiffEq v5.56.0
  [90014a1f] PDMats v0.11.0
  [65888b18] ParameterizedFunctions v5.10.0
  [d96e819e] Parameters v0.12.2
  [69de0a69] Parsers v1.1.0
  [ccf2f8ad] PlotThemes v2.0.1
  [995b91a9] PlotUtils v1.0.10
  [91a5bcdd] Plots v1.15.2
  [e409e4f3] PoissonRandom v0.4.0
  [f517fe37] Polyester v0.3.1
  [85a6dd25] PositiveFactorizations v0.2.4
  [21216c6a] Preferences v1.2.2
  [33c8b6b6] ProgressLogging v0.1.4
  [92933f4c] ProgressMeter v1.6.2
  [1fd47b50] QuadGK v2.4.1
  [67601950] Quadrature v1.8.1
  [8a4e6c94] QuasiMonteCarlo v0.2.2
  [74087812] Random123 v1.3.1
  [fb686558] RandomExtensions v0.4.3
  [e6cf234a] RandomNumbers v1.4.0
  [c84ed2f1] Ratios v0.4.0
  [3cdcf5f2] RecipesBase v1.1.1
  [01d81517] RecipesPipeline v0.3.2
  [731186ca] RecursiveArrayTools v2.11.4
  [f2c3362d] RecursiveFactorization v0.1.12
  [189a3867] Reexport v1.0.0
  [ae029012] Requires v1.1.3
  [ae5879a3] ResettableStacks v1.1.0
  [37e2e3b7] ReverseDiff v1.9.0
  [79098fc4] Rmath v0.7.0
  [7e49a35a] RuntimeGeneratedFunctions v0.5.2
  [476501e8] SLEEFPirates v0.6.20
  [1bc83da4] SafeTestsets v0.0.1
  [0bca4576] SciMLBase v1.13.4
  [30cb0354] SciMLTutorials v0.9.0
  [6c6a2e73] Scratch v1.0.3
  [efcf1570] Setfield v0.7.0
  [992d4aef] Showoff v1.0.3
  [699a6c99] SimpleTraits v0.9.3
  [ed01d8cd] Sobol v1.5.0
  [2133526b] SodiumSeal v0.1.1
  [b85f4697] SoftGlobalScope v1.1.0
  [a2af1166] SortingAlgorithms v1.0.0
  [47a9eef4] SparseDiffTools v1.13.2
  [684fba80] SparsityDetection v0.3.4
  [276daf66] SpecialFunctions v1.4.1
  [860ef19b] StableRNGs v1.0.0
  [aedffcd0] Static v0.2.4
  [90137ffa] StaticArrays v1.2.0
  [82ae8749] StatsAPI v1.0.0
  [2913bbd2] StatsBase v0.33.8
  [4c63d2b9] StatsFuns v0.9.8
  [9672c7b4] SteadyStateDiffEq v1.6.2
  [789caeaf] StochasticDiffEq v6.34.1
  [7792a7ef] StrideArraysCore v0.1.11
  [09ab397b] StructArrays v0.5.1
  [c3572dad] Sundials v4.4.3
  [d1185830] SymbolicUtils v0.11.2
  [0c5d862f] Symbolics v0.1.25
  [3783bdb8] TableTraits v1.0.1
  [bd369af6] Tables v1.4.2
  [5d786b92] TerminalLoggers v0.1.3
  [8290d209] ThreadingUtilities v0.4.4
  [f269a46b] TimeZones v1.5.5
  [a759f4b9] TimerOutputs v0.5.9
  [0796e94c] Tokenize v0.5.16
  [9f7883ad] Tracker v0.2.16
  [3bb67fe8] TranscodingStreams v0.9.5
  [592b5752] Trapz v2.0.2
  [a2a6695c] TreeViews v0.3.0
  [5c2747f8] URIs v1.3.0
  [3a884ed6] UnPack v1.0.2
  [1986cc42] Unitful v1.7.0
  [3d5dd08c] VectorizationBase v0.20.11
  [81def892] VersionParsing v1.2.0
  [19fa3120] VertexSafeGraphs v0.1.2
  [44d3d7a6] Weave v0.10.8
  [efce3f68] WoodburyMatrices v0.5.3
  [ddb6d928] YAML v0.4.6
  [c2297ded] ZMQ v1.2.1
  [a5390f91] ZipFile v0.9.3
  [e88e6eb3] Zygote v0.6.11
  [700de1a5] ZygoteRules v0.2.1
  [6e34b625] Bzip2_jll v1.0.6+5
  [83423d85] Cairo_jll v1.16.0+6
  [3bed1096] Cuba_jll v4.2.1+0
  [7bc98958] Cubature_jll v1.0.4+0
  [5ae413db] EarCut_jll v2.1.5+1
  [2e619515] Expat_jll v2.2.10+0
  [b22a6f82] FFMPEG_jll v4.3.1+4
  [f5851436] FFTW_jll v3.3.9+7
  [a3f928ae] Fontconfig_jll v2.13.1+14
  [d7e528f0] FreeType2_jll v2.10.1+5
  [559328eb] FriBidi_jll v1.0.5+6
  [0656b61e] GLFW_jll v3.3.4+0
  [d2c73de3] GR_jll v0.57.2+0
  [78b55507] Gettext_jll v0.21.0+0
  [7746bdde] Glib_jll v2.68.1+0
  [e33a78d0] Hwloc_jll v2.4.1+0
  [1d5cc7b8] IntelOpenMP_jll v2018.0.3+2
  [aacddb02] JpegTurbo_jll v2.0.1+3
  [c1c5ebd0] LAME_jll v3.100.0+3
  [dd4b983a] LZO_jll v2.10.1+0
  [dd192d2f] LibVPX_jll v1.9.0+1
  [e9f186c6] Libffi_jll v3.2.2+0
  [d4300ac3] Libgcrypt_jll v1.8.7+0
  [7e76a0d4] Libglvnd_jll v1.3.0+3
  [7add5ba3] Libgpg_error_jll v1.42.0+0
  [94ce4f54] Libiconv_jll v1.16.1+0
  [4b2f31a3] Libmount_jll v2.35.0+0
  [89763e89] Libtiff_jll v4.1.0+2
  [38a345b3] Libuuid_jll v2.36.0+0
  [856f044c] MKL_jll v2021.1.1+1
  [e7412a2a] Ogg_jll v1.3.4+2
  [458c3c95] OpenSSL_jll v1.1.1+6
  [efe28fd5] OpenSpecFun_jll v0.5.4+0
  [91d4177d] Opus_jll v1.3.1+3
  [2f80f16e] PCRE_jll v8.44.0+0
  [30392449] Pixman_jll v0.40.1+0
  [ea2cea3b] Qt5Base_jll v5.15.2+0
  [f50d1b31] Rmath_jll v0.3.0+0
  [fb77eaff] Sundials_jll v5.2.0+1
  [a2964d1f] Wayland_jll v1.17.0+4
  [2381bf8a] Wayland_protocols_jll v1.18.0+4
  [02c8fc9c] XML2_jll v2.9.12+0
  [aed1982a] XSLT_jll v1.1.34+0
  [4f6342f7] Xorg_libX11_jll v1.6.9+4
  [0c0b7dd1] Xorg_libXau_jll v1.0.9+4
  [935fb764] Xorg_libXcursor_jll v1.2.0+4
  [a3789734] Xorg_libXdmcp_jll v1.1.3+4
  [1082639a] Xorg_libXext_jll v1.3.4+4
  [d091e8ba] Xorg_libXfixes_jll v5.0.3+4
  [a51aa0fd] Xorg_libXi_jll v1.7.10+4
  [d1454406] Xorg_libXinerama_jll v1.1.4+4
  [ec84b674] Xorg_libXrandr_jll v1.5.2+4
  [ea2f1a96] Xorg_libXrender_jll v0.9.10+4
  [14d82f49] Xorg_libpthread_stubs_jll v0.1.0+3
  [c7cfdc94] Xorg_libxcb_jll v1.13.0+3
  [cc61e674] Xorg_libxkbfile_jll v1.1.0+4
  [12413925] Xorg_xcb_util_image_jll v0.4.0+1
  [2def613f] Xorg_xcb_util_jll v0.4.0+1
  [975044d2] Xorg_xcb_util_keysyms_jll v0.4.0+1
  [0d47668e] Xorg_xcb_util_renderutil_jll v0.3.9+1
  [c22f9ab0] Xorg_xcb_util_wm_jll v0.4.1+1
  [35661453] Xorg_xkbcomp_jll v1.4.2+4
  [33bec58e] Xorg_xkeyboard_config_jll v2.27.0+4
  [c5fb5394] Xorg_xtrans_jll v1.4.0+3
  [8f1865be] ZeroMQ_jll v4.3.2+6
  [3161d3a3] Zstd_jll v1.5.0+0
  [0ac62f75] libass_jll v0.14.0+4
  [f638f0a6] libfdk_aac_jll v0.1.6+4
  [b53b4c65] libpng_jll v1.6.38+0
  [a9144af2] libsodium_jll v1.0.20+0
  [f27f6e37] libvorbis_jll v1.3.6+6
  [1270edf5] x264_jll v2020.7.14+2
  [dfaa095f] x265_jll v3.0.0+3
  [d8fb68d0] xkbcommon_jll v0.9.1+5
  [0dad84c5] ArgTools
  [56f22d72] Artifacts
  [2a0f44e3] Base64
  [ade2ca70] Dates
  [8bb1440f] DelimitedFiles
  [8ba89e20] Distributed
  [f43a241f] Downloads
  [7b1f6079] FileWatching
  [9fa8497b] Future
  [b77e0a4c] InteractiveUtils
  [4af54fe1] LazyArtifacts
  [b27032c2] LibCURL
  [76f85450] LibGit2
  [8f399da3] Libdl
  [37e2e46d] LinearAlgebra
  [56ddb016] Logging
  [d6f4376e] Markdown
  [a63ad114] Mmap
  [ca575930] NetworkOptions
  [44cfe95a] Pkg
  [de0858da] Printf
  [9abbd945] Profile
  [3fa0cd96] REPL
  [9a3f8284] Random
  [ea8e919c] SHA
  [9e88b42a] Serialization
  [1a1011a3] SharedArrays
  [6462fe0b] Sockets
  [2f01184e] SparseArrays
  [10745b16] Statistics
  [4607b0f0] SuiteSparse
  [fa267f1f] TOML
  [a4e569a6] Tar
  [8dfed614] Test
  [cf7118a7] UUIDs
  [4ec0a83e] Unicode
  [e66e0078] CompilerSupportLibraries_jll
  [deac9b47] LibCURL_jll
  [29816b5a] LibSSH2_jll
  [c8ffd9c3] MbedTLS_jll
  [14a3606d] MozillaCACerts_jll
  [4536629a] OpenBLAS_jll
  [bea87d4a] SuiteSparse_jll
  [83775a58] Zlib_jll
  [8e850ede] nghttp2_jll
  [3f19e933] p7zip_jll
```

