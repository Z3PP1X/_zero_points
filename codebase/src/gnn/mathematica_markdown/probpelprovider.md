(* ::Package:: *)

BeginPackage["ProblemProvider`"];


(* =========================================================
   Public API
   gmgfSolver.wl

   gMGF root finder
   with timing and convergence analysis.
   functional pipeline implementation.

   Author:   Patrick Fuchs | p.fuchs@hm.edu
   Version:  1.0.0
   Requires: Wolfram Language 14+
   ========================================================= *)


InitQueue::usage = "InitQueue[list, n] initialisiert die Registry und f\[UDoubleDot]llt die Queue mit n Startproblemen.";
FetchNext::usage = "FetchNext[] liefert den n\[ADoubleDot]chsten State aus der Queue (f\[UDoubleDot]r den PipelineWorker).";
GetProblemfunction::usage = "GetProblemfunction[id] liefert die Pure function zu einer ID (f\[UDoubleDot]r die Solver).";
PushToDataQueue::usage = "Queue a problem in the dataQueue";
$dataPattern::usage = "Global Datapattern for ML-Pipeline States";
$functionRegistry::usage = "Function registry for used problems!";
$problemList::usage ="Global problemList";

GetBenchmarkTime::usage = "GetBenchmarkTime[solver, id, yTarget] looks up a cached \
benchmark wall-clock time for the *opposing* solver on (id, yTarget). Returns Missing[] if unknown.";
SetBenchmarkTime::usage = "SetBenchmarkTime[solver, id, yTarget, t] stores a benchmark time. \
solver is the OPPOSING solver (the one being timed: 0=Newton, 1=gMGF).";
HasBenchmarkTime::usage = "HasBenchmarkTime[solver, id, yTarget] returns True if a cached \
benchmark time exists.";


SetProblemList::usage = "SetProblemList[list] \[UDoubleDot]berschreibt die interne Problem-Liste.";
AddToQueue::usage = "AddToQueue[data] pusht einen State manuell in die Problem-Queue.";


Begin["`Private`"]


$dataQueue = CreateDataStructure["Queue"];
$functionRegistry = CreateDataStructure["HashTable"];
$benchmarkRegistry = CreateDataStructure["HashTable"];

benchmarkKey[solver_, id_, yTarget_] := {solver, id, N[yTarget]};

GetBenchmarkTime[solver_, id_, yTarget_] := Module[{key, val},
    key = benchmarkKey[solver, id, yTarget];
    val = $benchmarkRegistry["Lookup", key];
    If[MissingQ[val], Missing[], val]
];

HasBenchmarkTime[solver_, id_, yTarget_] := 
    $benchmarkRegistry["KeyExistsQ", benchmarkKey[solver, id, yTarget]];

SetBenchmarkTime[solver_, id_, yTarget_, t_?NumericQ] := (
    $benchmarkRegistry["Insert", benchmarkKey[solver, id, yTarget] -> t];
    t
);

$problemList = {
<| "id" -> "P1", "function" -> Function[x, x*Exp[x^2] - Sin[x]^2 + 3 Cos[x] + 5],
     "x0" -> 0.0, "yRange" -> {-10.0, 8.0} |>,
  <| "id" -> "P2",
     "function" -> Function[x, x*Exp[x^2] - Sin[x]^2 + 3 Cos[x] + 5],
     "x0" -> 1.0, "yRange" -> {8.1, 100.0} |>,
  <| "id" -> "P3",
     "function" -> Function[x, x^(1/3)*(x - Exp[x])],
     "x0" -> 0.5, "yRange" -> {-80.0, -0.5} |>,
  (*<| "id" -> "P4", 
     "function" -> Function[x, (1/x^2) + (10/x^4) + (100/x^10)],
     "x0" -> 2.5, "yRange" -> {0.1, 10000.0} |>, *)
  <| "id" -> "P5",
     "function" -> Function[x, -(1/x) + (1/Sqrt[x]) + ((3/20)*x^10)],
     "x0" -> 1.0, "yRange" -> {0.1, 100.0} |>,
  <| "id" -> "P6", 
     "function" -> Function[x, x^9 + x^7 + x^2],
     "x0" -> 0.2, "yRange" -> {0.1, 99.1} |>
};

isfunctionQ[f_] := Quiet[NumericQ[f[1.0]]]

$dataPattern = Association[
	"id" -> (_String | _Integer),
	"status" -> ("init" | "training" | "reward_calc" | "finished"),
	"stateId" -> _String,
	"uuid" -> _String,
	"parentStateId" -> (_String | Symbol["Null"]),
	"networkJobId" -> _String,
	"inRewardCalc" -> (1 | 0),
	"networkStep" -> _Integer,
	"epoch" -> _Integer,
	"function" -> (_String | _Integer),
	"tolerance" -> _?Positive,
	"maxIter" -> _?Positive,
	"yMin" -> _?NumericQ,
	"currentX" -> _?NumericQ,
    "iterSteps" -> _?NumericQ,
	"yMax" -> _?NumericQ,
	"yTarget" -> _?NumericQ,
	"timeBenchmarkSolver" -> _?NumericQ,
	"x0" -> _?NumericQ,
	"mask" -> (1 | 0),
	"solver" -> (1 | 0),
	"absTime" -> _?NumericQ,
	"recordAbsTime" -> _?NumericQ,
	"localMaxTolerance" -> _?NumericQ,
	"lastStepError" -> _?NumericQ,
	"fx" -> _?NumericQ,
	"dfx" -> _?NumericQ,
	"ddfx" -> _?NumericQ,
	"kappa" -> _Integer,
	"lastKappa" -> _Integer
];

SetProblemList[list_List] := (
    $problemList = list;    
    Scan[$functionRegistry["Insert", #["id"] -> #["function"]] &, list];
);

PushToDataQueue[data_Association] := (
	$dataQueue["Push", data]
)

GetProblemfunction[id_String] := $functionRegistry["Lookup", id];


(*
  setInitStateConfig: build the initial state for a (problem, yTarget) pair.
  Each state gets a fresh ``CreateUUID[]`` to keep the Python replay buffer
  keyed uniquely per episode, even across epochs of the same problem.

  ``recordAbsTime`` is initialized to a large sentinel (1.0e9 seconds) so the
  first epoch's ``r_learn`` term doesn't see a fake "record" of 1 second.
  Python's RewardCalculator treats values above ``record_abs_time_sentinel``
  as "no record yet" (see reward.py).
*)
setInitStateConfig[data_Association, yMin_?NumericQ, yMax_?NumericQ] := Module[{
    id = data["id"],
    yT = data["yTarget"],
    ep = 0,
    ns = 0,
    x = data["x0"]  
    },
    Join[data, <|
        "status" -> "init",
        "initialx0" -> data["x0"],
        "uuid" -> CreateUUID[],
        "epoch" -> ep,
        "networkStep" -> ns,
        "parentStateId" -> "initialState",
        "networkJobId" -> "init",
        "tolerance" -> 0.000000000000001,
        "inRewardCalc" -> 0,
        "maxIter" -> 1500,
        "yMin" -> yMin,
        "yMax" -> yMax,
        "timeBenchmarkSolver" -> 0,
        "stateId" -> TemplateApply["``_yTarget_``_epoch_``_networkstep_``", {id, yT, ep, ns}],
        "mask" -> 1,
        "solver" -> 1,
        "absTime" -> 0,
        "recordAbsTime" -> 1.0*^9,
        "currentX" -> data["x0"],
        "iterSteps" -> 0,
        "localMaxTolerance" -> 1.0,
        "lastStepError" -> 0.0,
        "fx" -> data["x0"],
        "dfx" -> 0.0,
        "ddfx" -> 0.0,
        "kappa" -> 0,
        "lastKappa" -> 0
    |>]
]

GenerateAndQueueProblems[problemList_List, dataPoints_Integer] := Module[
    {allProblems},
    allProblems = Catenate[
        Function[prob, 
            Module[{
                strippedProb = Append[KeyDrop[prob, "function"], "function" -> prob["id"]]
            },
                setInitStateConfig[
                    Join[strippedProb, <|"yTarget" -> N[#]|>],
                    prob["yRange"][[1]],                
                    prob["yRange"][[2]]                
                ] & /@ Subdivide[prob["yRange"][[1]], prob["yRange"][[2]], Max[1, dataPoints - 1]]
            ]
        ] /@ problemList
    ];
    Scan[AddToQueue, RandomSample[allProblems]];
    Print["[SYSTEM: Queue Ready!]"]
];
	
AddToQueue[data_Association] := $dataQueue["Push", data];

FetchNext[] := If[$dataQueue["EmptyQ"],
	$Failed,
	$dataQueue["Pop"]
	];

InitQueue[list_List, n_Integer:1000] := (
    SetProblemList[list];
    GenerateAndQueueProblems[list, n]
);







End[];


EndPackage[]
