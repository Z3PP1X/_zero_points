(* ::Package:: *)

BeginPackage["gmgfSolver`"];


(* =========================================================
   Public API
   gmgfSolver.wl

   gMGF root finder
   with timing and convergence analysis.
   functional pipeline implementation.

   Author:   Patrick Fuchs | p.fuchs@hm.edu
   Version:  2.1.3
   Requires: Wolfram Language 14+
   ========================================================= *)
Needs["ProblemProvider`"];
Needs["newtonSolver`"];

GetGMGFResult::usage = "GetGMGFResult[config] runs a Generalised Modified Gradient \
Flow root-finding job defined by config and returns an Association with the result, \
convergence info, and timing.


config must be an Association with keys:
  \"id\"            \[Dash] String or Integer identifier,
  \"compile\"       \[Dash] Boolean,
  \"function\"      \[Dash] a numeric pure function f,
  \"x0\"            \[Dash] initial guess (numeric),
  \"yRange\"        \[Dash] {yMin, yMax} defining the domain via f(x) in [yMin, yMax],
  \"tolerance\"     \[Dash] positive real convergence threshold on step gain,
  \"maxIterations\" \[Dash] positive integer iteration cap.

The solver builds a kappa lookup table from f, f', f'' to select the
appropriate H-transform at each step, then iterates x_(n+1) = H[x] / f'[x]
until |step_gain| < tolerance or maxIterations is reached.

Returns an Association with keys \"job_id\", \"result\", and \"analysis\".
Returns $Failed on non-convergence or zero derivative.";
setUpgMGFConnection::usage = "";
RemoveFinishedfromActiveRegistry::usage = "Remove all Pure functions (K-Lookups, H-functions, fx, dfx...) from the $ActiveRegistry for a state that has reached reward_calc"


Begin["`Private`"];


$gmgftaskReceiver = None;
$gmgfresultsProvider = None;
$gmgflistenTask = None;

parseMessage[event_] := Module[{jsonString, jsonObject, funcExpr},
    Print["[DAEMON] Empfangenes Event-Objekt: ", Keys[event]];
    
    If[KeyExistsQ[event, "DataByteArray"],
        jsonString = ByteArrayToString[event["DataByteArray"], "UTF-8"];
    ,
        jsonString = Lookup[event, "Data", ""];
    ];
    If[!StringQ[jsonString] || StringLength[StringTrim[jsonString]] == 0,
        Print["[\|01f6a8 DAEMON FEHLER] Der empfangene String ist leer oder kein Text!"];
        Return[<|"id" -> "UNKNOWN", "status" -> "error", "errorMessage" -> "Empty Socket Data"|>];
    ];
    jsonObject = Quiet @ Check[Association @ ImportString[jsonString, "JSON"], $Failed];
    If[jsonObject === $Failed,
        Print["[\|01f6a8 DAEMON FEHLER] Das JSON konnte nicht geparst werden!"];
        Return[<|"id" -> "UNKNOWN", "status" -> "error", "errorMessage" -> "Invalid JSON"|>];
    ];
    If[KeyExistsQ[jsonObject, "function"] && StringQ[jsonObject["function"]],
        funcExpr = Quiet @ Check[ToExpression[jsonObject["function"]], $Failed];
        If[funcExpr === $Failed,
            Print["[\|01f6a8 DAEMON FEHLER] Funktion konnte nicht kompiliert werden: ", jsonObject["function"]];
            Return[<|"id" -> Lookup[jsonObject, "id", "UNKNOWN"], "status" -> "error", "errorMessage" -> "Invalid Math Function"|>];
        ];
        jsonObject["function"] = funcExpr;
    ];
    jsonObject   
]

jobPush[state_Association] := Module[{jsonString, bytes},
	Echo[state, "State vor Export"];
	jsonString = ExportString[state, "JSON", "Compact" -> True];
	jsonString = StringDelete[jsonString, {"\n", "\r"}];
	bytes = StringToByteArray[jsonString, "UTF-8"]; 
	BinaryWrite[$gmgfresultsProvider, bytes];
];

catchSolverErrors[state_Association] := Module[{evalData},
  Print["[NEWTON-DAEMON] Starte Newton-Solver f\[UDoubleDot]r State ID: ", Lookup[state, "id", "UNKNOWN"]];
  evalData = EvaluationData[
    GetGMGFResult[state]
  ];
  If[Length[evalData["MessagesText"]] > 0,
    Print["\n[\|01f6a8 NEWTON SOLVER CRASH / WARNUNG \|01f6a8]"];
    Print[StringRiffle[evalData["MessagesText"], "\n"]];
    Print["[\|01f6a8 ============================== \|01f6a8]\n"];
  ];
  If[!evalData["Success"],
    Return[<|
      "id" -> Lookup[state, "id", "UNKNOWN"], 
      "status" -> "error", 
      "errorMessage" -> evalData["MessagesText"]
    |>]
  ];
  evalData["Result"]
]


setUpgMGFConnection[portIN_, portOUT_] := (
	$gmgftaskReceiver = SocketConnect[portIN, "ZMQ_PULL"];
	$gmgfresultsProvider = SocketOpen[portOUT, "ZMQ_PUSH"];
	ProblemProvider`SetProblemList[ProblemProvider`$problemList];
	
	Print["[GMGF-DAEMON] Sockets connected, listening for Events..."];
	
	$listenTask = SocketListen[$gmgftaskReceiver, parseMessage /* catchSolverErrors /* jobPush];
	
	
)


$ActiveRegistry = CreateDataStructure["HashTable"];
$ActiveStateKeys = {
    "f", "df", "ddf", "intervals", "kappaList", 
    "hList", "findIndex", "getKappa", "getH", "kappaLookup"
};

(* --------------------------------------------------------- *)
(* Helper Predicates                                         *)
(* --------------------------------------------------------- *)

removeFromActiveRegistry[state_Association] := Module[{cacheKey},
  If[state["status"] === "reward_calc",
    cacheKey = {state["id"], state["yTarget"]};
    $ActiveRegistry["KeyDrop", cacheKey];
  ];
  state
]

isfunctionQ[f_] := Quiet[NumericQ[f[1.0]]]

isRealNumericQ[val_] := NumericQ[val] && Im[val] == 0

hydrate[state_] := <|state, "function" -> ProblemProvider`GetProblemfunction[state["id"]]|>;

ensureInitialized[state_?AssociationQ] := Module[{cacheKey, cachedData, requiredKeys},
  If[AnyTrue[$ActiveStateKeys, MissingQ[state[#]] &],
    
    cacheKey = {state["id"], state["yTarget"]};
    cachedData = $ActiveRegistry["Lookup", cacheKey];
    
    If[MissingQ[cachedData],
      With[{newState = state // init // buildKappaLookup},
        $ActiveRegistry["Insert", cacheKey -> KeyTake[newState, 
          {"f", "df", "ddf", "intervals", "kappaList", "hList", "findIndex", "getKappa", "getH"}
        ]];
        newState
      ],
      Join[state, cachedData, <|"kappaLookup" -> True|>]
    ],
    state
  ]
]

ensureMaskInactive[state_?AssociationQ] := MapAt[
  If[# === 1, 0, #] &, 
  state, 
  {"mask"}
]

dehydrate[state_] := KeyDrop[
    <|state, "function" -> state["id"]|>, 
    {
      "f", "df", "ddf", "intervals", "kappaList", 
      "hList", "findIndex", "getKappa", "getH", "kappaLookup"
    }
];

(* --------------------------------------------------------- *)
(* 2. Core Algorithm (Math & H/Kappa)                        *)
(* --------------------------------------------------------- *)

buildHfunction[0]                  := Identity
buildHfunction[k_Integer?Positive] := Function[var, Nest[Exp[#] - 1 &, var, k]]
buildHfunction[k_Integer?Negative] := Function[var, Nest[Log[# + 1] &, var, Abs[k]]]

gMGFAlgo[f_, df_, x0_, getH_, tolerance_, maxIter_] := Module[
  {error = 100, xnew, xold = x0, initX = x0, fx, dfx, Hfunction, iter = 0},
  While[error > tolerance && iter < maxIter,
    fx  = f[xold];
    dfx = df[xold];

    Hfunction = getH[xold];
    xnew  = xold - Sign[fx] * (Hfunction[Abs[fx]] / dfx);
    error = Abs[xnew - xold];
    xold  = xnew;
    iter++
  ];
  <|
    "x" -> xold, 
    "fx" -> fx, 
    "dfx" -> dfx, 
    "initX" -> initX, 
    "error" -> error, 
    "iterations" -> iter
  |>
]

buildKappaLookup[state_?AssociationQ] :=
  If[!MissingQ[state["kappaLookup"]],
    state,
    With[{
        f    = state["f"],
        df   = state["df"],
        ddf  = state["ddf"],
        yT   = state["yTarget"],
        x0   = state["currentX"],
        yMin = state["yMin"],
        yMax = state["yMax"]
    },
    With[{
        r1 = Quiet @ FindRoot[f[x] == yMin - yT, {x, x0}],
        r2 = Quiet @ FindRoot[f[x] == yMax - yT, {x, x0}]
    },
    With[{
        x1 = If[MatchQ[r1, {(_Rule | _RuleDelayed) ..}], x /. r1, x0 - 1.0],
        x2 = If[MatchQ[r2, {(_Rule | _RuleDelayed) ..}], x /. r2, x0 + 1.0]
    },
    With[{
        xMin = Min[x1, x2] - 0.001,
        xMax = Max[x1, x2] + 0.001
    },
    With[{
        kappaCont = Function[t, Sign[f[t]] * ddf[t] / df[t]^2]
    },
    With[{
        samples = Select[
            Table[
                {t, Round[Quiet[kappaCont[t]]]},
                {t, xMin, xMax, (xMax - xMin) / 1000.}
            ],
            NumericQ[#[[2]]] && -25 <= #[[2]] <= 25 &
        ]
    },
    With[{
        runs = Split[samples, #1[[2]] == #2[[2]] &]
    },
    With[{
        finalTable = Map[
            Function[{run},
                With[{
                    k   = run[[1, 2]],
                    x1g = Min[run[[All, 1]]],
                    x2g = Max[run[[All, 1]]]
                },
                With[{
                    x1p = Quiet @ Check[
                        x /. FindRoot[kappaCont[x] - (k - 0.5), {x, x1g}],
                        x1g
                    ],
                    x2p = Quiet @ Check[
                        x /. FindRoot[kappaCont[x] - (k + 0.5), {x, x2g}],
                        x2g
                    ]
                },
                    {k, Min[x1p, x1g], Max[x2p, x2g]}
                ]]
            ],
            runs
        ]
    },
    With[{
        clipped = Module[{ft = finalTable},
            Do[
                If[ft[[i, 3]] > ft[[i + 1, 2]],
                    With[{mid = (ft[[i, 3]] + ft[[i + 1, 2]]) / 2.0},
                        ft[[i, 3]]     = mid;
                        ft[[i + 1, 2]] = mid;
                    ]
                ],
                {i, Length[ft] - 1}
            ];
            ft
        ]
    },
    With[{
        intervals = clipped[[All, {2, 3}]],
        kappaList = clipped[[All, 1]],
        starts    = clipped[[All, 2]]
    },
    With[{
        hList = Map[buildHfunction, kappaList]
    },
    With[{
        kl = kappaList,
        hl = hList,
        s  = starts,
        n  = Length[starts]
    },
    With[{
        findIdx = Function[xVal,
            With[{pos = Total[UnitStep[xVal - s]]},
                Clip[pos, {1, n}]
            ]
        ]
    },
        Join[state, <|
            "kappaLookup" -> True,
            "status"      -> "training",
            "intervals"   -> intervals,
            "kappaList"   -> kl,
            "hList"       -> hl,
            "findIndex"   -> findIdx,
            "getKappa"    -> Function[v, kl[[findIdx[v]]]],
            "getH"        -> Function[v, hl[[findIdx[v]]]]
        |>]
    ]]]]]]]]]]]]]]



(* --------------------------------------------------------- *)
(* 3. State Management & Pipeline Steps                      *)
(* --------------------------------------------------------- *)
(* Deprecated?? init[config_Association] := With[{
    id            = config["id"],
    stateId       = config["stateId"],
    status        = config["status"],
    parentStateId = config["parentStateId"],
    networkStep   = config["networkStep"],
    function      = config["function"],
    x0            = config["x0"],
    yMin          = config["yMin"],
    yMax          = config["yMax"],
    yTarget       = config["yTarget"],
    absTime       = config["absTime"],
    currentX      = config["currentX"],
    iterSteps     = config["iterSteps"],
    tol           = Lookup[config, "tolerance",         1.*^-15],
    lMaxtol       = Lookup[config, "localMaxTolerance", 0.0],
    maxI          = Lookup[config, "maxIter",           1500]
},
    <|
        "id"                -> id,
        "stateId"           -> stateId,
        "status"            -> status,
        "parentStateId"     -> parentStateId,
        "networkStep"       -> networkStep,
        "tolerance"         -> tol,
        "localMaxTolerance" -> lMaxtol,
        "maxIter"           -> maxI,
        "yMin"              -> yMin,
        "yMax"              -> yMax,
        "x0"                -> x0,
        "yTarget"           -> yTarget,
        "absTime"           -> absTime,
        "currentX"          -> currentX,
        "iterSteps"          -> iterSteps,
        "iterSteps"         -> Lookup[config, "iterSteps",      0],
        "absTime"           -> Lookup[config, "absTime",        0.0],
        "lastStepError"     -> Lookup[config, "lastStepError",  1],
        "f"                 -> (function[#] -yTarget &),
        "df"                -> Derivative[1][function],
        "ddf"               -> Derivative[2][function],
        "kappa"             -> Lookup[config, "kappa",         0],
        "lastKappa"         -> Lookup[config, "lastKappa",     0],
        "kappaLookup"       -> Missing[]
    |>
]
*)
init[config_Association] := Module[{function, yTarget},
    function = config["function"];
    yTarget = config["yTarget"];
    
    Join[
        config, 
        <|
            "tolerance"         -> Lookup[config, "tolerance", 1.*^-15],
            "localMaxTolerance" -> Lookup[config, "localMaxTolerance", 0.0],
            "maxIter"           -> Lookup[config, "maxIter", 1500],
            "iterSteps"         -> Lookup[config, "iterSteps", 0],
            "absTime"           -> Lookup[config, "absTime", 0.0],
            "lastStepError"     -> Lookup[config, "lastStepError", 1.0],
            "f"                 -> (function[#] - yTarget &),
            "df"                -> Derivative[1][function],
            "ddf"               -> Derivative[2][function]
        |>
    ]
]        



gMGFStep[state_?AssociationQ] :=
  If[state["status"] =!= "training",
    state,
    Module[{timing, result, startX},
      startX = state["currentX"];
      timing = AbsoluteTiming[
        gMGFAlgo[
          state["f"],
          state["df"],
          startX,
          state["getH"],
          state["localMaxTolerance"],
          state["maxIter"]
        ]
      ];
      result = timing[[2]];
      
      <|state,
        "absTime"       -> state["absTime"] + timing[[1]],
        "currentX"      -> result["x"],
        "fx"            -> result["fx"],
        "dfx"           -> result["dfx"],
        "lastStepError" -> result["error"],
        "iterSteps"     -> state["iterSteps"] + result["iterations"],
        "kappa"         -> state["getKappa"][result["x"]],
        "lastKappa"     -> state["getKappa"][startX]
      |>
    ]
  ]
  
updateStatus[state_?AssociationQ] :=
  If[state["status"] === "training" &&
     Abs[state["lastStepError"]] < state["tolerance"],
    <|state, "status" -> "reward_calc"|>,
    state
  ]

rotateStateIds[state_?AssociationQ] := <|state,
  "parentStateId" -> state["stateId"],
  "stateId" -> StringTemplate["`1`_yTarget_`2`_epoch_`3`_networkstep_`4`"][
    state["id"], 
    state["yTarget"], 
    state["epoch"],
    state["networkStep"]
  ]
|>

(*** Deprecated  
checkConvergence[state_?AssociationQ] :=
  If[Abs[state["step_gain"]] <= state[["settings", "tolerance"]],
    state,
    Throw[
      <|
        "job_id"   -> state["id"],
        "result"   -> <|
            "zero_point"  -> state["currentX"],
            "iterations"  -> state["iterSteps"],
            "gmgf_time_s" -> -1.0
        |>,
        "analysis" -> <|
            "converged"  -> False,
            "last_error" -> state["step_gain"]
        |>
      |>,
      "nonConvergence"
    ]
  ]

finalise[state_?AssociationQ] := <|
  "job_id"   -> state["id"],
  "result"   -> <|
      "zero_point"  -> state["currentX"],
      "iterations"  -> state["iterSteps"],
      "gmgf_time_s" -> state["absTime"]
  |>,
  "analysis" -> <|
      "converged"  -> (Abs[state["step_gain"]] <= state[["settings", "tolerance"]]),
      "last_error" -> state["step_gain"]
  |>
|> --- *)
changetoTraining[state_Association] := 
  If[state["status"] === "init", 
    <| state, "status" -> "training" |>, 
    state
  ]

bumpNetworkStep[state_?AssociationQ] := <|state,
  "networkStep" -> state["networkStep"] + 1
|>

removeFromActiveRegistry[state_Association] := (
    If[state["status"] === "reward_calc", 
        $ActiveRegistry["KeyDrop", state["id"]]
    ];
    state
)

resetRegistry[state_] := (
	$ActiveRegistry = CreateDataStructure["HashTable"];
	state
)



(* BENCHMARK *)

$HelperRegistry = CreateDataStructure["HashTable"];
$newtonActiveStateKeys = {"f", "df", "ddf"};

newtonresetRegistry[state_] := (
	$HelperRegistry = CreateDataStructure["HashTable"];
	state
)

newtoninit[config_Association] := Module[{function, yTarget},
    function = config["function"];
    yTarget = config["yTarget"];
    
    Join[
        config, 
        <|
            "tolerance"         -> Lookup[config, "tolerance", 1.*^-15],
            "localMaxTolerance" -> Lookup[config, "localMaxTolerance", 0.0],
            "maxIter"           -> Lookup[config, "maxIter", 1500],
            "iterSteps"         -> Lookup[config, "iterSteps", 0],
            "absTime"           -> Lookup[config, "absTime", 0.0],
            "lastStepError"     -> Lookup[config, "lastStepError", 1.0],
            "df"                -> Lookup[config, "df", Derivative[1][function]],
            "ddf"               -> Lookup[config, "df", Derivative[2][function]]
        |>
    ]
]

newtonensureInitialized[state_?AssociationQ] := Module[{cacheKey, cachedData},
  If[AnyTrue[$newtonActiveStateKeys, MissingQ[state[#]] &],
    cacheKey = {state["id"], Lookup[state, "yTarget", state["yTarget"]]};
    cachedData = $HelperRegistry["Lookup", cacheKey];
    
    If[MissingQ[cachedData],
      With[{newState = state // newtoninit},
        $HelperRegistry["Insert", cacheKey -> KeyTake[newState, {"f", "df", "ddf"}]];
        newState
      ],
      Join[state, cachedData]
    ],
    state
  ]
]
  
newtonAlgo[f_, df_, x0_, tolerance_, maxIter_] := Module[
  {error = 100, xnew, xold = x0, initX = x0, fx, dfx, iter = 0},
  While[error > tolerance && iter < maxIter,
    fx  = f[xold];
    dfx = df[xold];
    
    If[dfx == 0 || !isRealNumericQ[dfx], Break[]];

    xnew  = xold - (fx / dfx);
    error = Abs[xnew - xold];
    xold  = xnew;
    iter++
  ];
  <|
    "x" -> xold, 
    "fx" -> fx, 
    "dfx" -> dfx, 
    "initX" -> initX, 
    "error" -> error, 
    "iterations" -> iter
  |>
]

newtonStep[state_?AssociationQ] :=
  If[state["status"] =!= "training",
    state,
    Module[{timing, result, startX},
      startX = state["currentX"];
      
      timing = AbsoluteTiming[
        newtonAlgo[
          state["f"],   (* Nutzt direkt die geladene Funktion *)
          state["df"],  (* Nutzt die geladene Ableitung *)
          startX,
          state["localMaxTolerance"],
          state["maxIter"]
        ]
      ];
      result = timing[[2]];
      
      <|state,
        "absTime"       -> state["absTime"] + timing[[1]],
        "currentX"      -> result["x"],
        "fx"            -> result["fx"],
        "dfx"           -> result["dfx"],
        "ddfx"          -> state["ddf"][result["x"]], (* Zugriff auf ddf im State *)
        "lastStepError" -> result["error"],
        "iterSteps"     -> state["iterSteps"] + result["iterations"]
      |>
    ]
  ]
  
newtonbenchmark[state_] := With[{
    benchstate = NewtonBenchmarkResult[Append[state, "absTime" -> 0.0]]
  },
    Append[state, "timeBenchmarkSolver" -> benchstate["absTime"]]
]
  
NewtonBenchmarkResult[config_?AssociationQ] := 
    config //
    newtonresetRegistry //
    newtonensureInitialized //
    newtonStep


EchoBeforeBenchmark[state_] := (Echo[state,"Before Benchmark: "]; state)
EchoAfterBenchmark[state_] := (Echo[state, "After Benchmark: "]; state)
EchoAfterStep[state_] := (Echo[state, "After Step, "]; state)


GetGMGFResult[config_?AssociationQ] := 
    config //
    ensureMaskInactive //
    hydrate //
    ensureInitialized //
    rotateStateIds    //
    changetoTraining  //
    bumpNetworkStep   //
    EchoBeforeBenchmark //
    NewtonBenchmarkResult //
    EchoAfterBenchmark //
    gMGFStep          //
    EchoAfterStep    //
    updateStatus      //
    removeFromActiveRegistry //
    dehydrate


End[];


EndPackage[];
