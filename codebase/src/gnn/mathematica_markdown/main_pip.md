(* ::Package:: *)

ClearAll["Global`*"];
ClearSystemCache[];
SetDirectory[NotebookDirectory[]];
If[!MemberQ[$Path, NotebookDirectory[]], AppendTo[$Path, NotebookDirectory[]]];

Get["ProblemProvider.wl"];
Get["newtonSolver.wl"];
Get["gmgfSolver.wl"];

Print["Mathematica ", $Version];
Print["Dependencies Successfully Loaded ", DateString[]];



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
     "x0" -> 0.2, "yRange" -> {0.1, 99.1} |>,
  <| "id" -> "P7", 
     "function" -> Function[x, Exp[-x] + (x/5)],
     "x0" -> 1.0, "yRange" -> {0.1, 99.0} |>
     
};


runName = "run_" <> DateString[{"Year","Month","Day","_","Hour","Minute","Second"}];
runsBaseDir = FileNameJoin[{NotebookDirectory[], "runs"}];
If[!DirectoryQ[runsBaseDir], CreateDirectory[runsBaseDir]];

runDir = FileNameJoin[{runsBaseDir, runName}];
CreateDirectory[runDir];
logFilePath = FileNameJoin[{runDir, "main_log.jsonl"}];
Print["\n=== New Run Initialised ==="];
Print["Run-Directory: ", runDir];

$batchSize = 32;
$jobsPerProblem = 1000;

ProblemProvider`InitQueue[$problemList, $jobsPerProblem];


(* DAEMON Setup and Pipeline Queues*)


$maxEpoch = 1000;
$activeRunningProblems = CreateDataStructure["HashSet"];
$activeProblems = CreateDataStructure["Queue"];
$rewardQueue = CreateDataStructure["Queue"];

$pipelineTask = None; 
$currentLogStream = None;
$failedLogStream = None;

$nnProvider = None;
$nnSink = None;
$taskProvider = None;

$newtonProvider = None;
$newtonSink = None;
$gmgfProvider = None;
$gmgfSink = None;

$resultsProvider = None;
$rewardPush = None;
$rewardSink = None;
$rewardSinkListen = None;

$calculatedRewardPush = None;
$calculatedRewardSink = None;
$calculatedRewardSinkListen = None;
$rewardPush = None;

$nnProviderPORT = 5650;
$nnSinkPORT = 5651;

$gmgfTaskPORT = 5670;
$gmgfResPORT = 5671;

$newtonTaskPORT = 5680;
$newtonResPORT = 5681;

$rewardPushPORT = 5690;
$calculatedRewardPushPORT = 5692;
$resultsPORT = 5693;

$running = True; 


startPipelineTask[] := (
    If[$pipelineTask =!= None, TaskRemove[$pipelineTask]];
    $pipelineTask = SessionSubmit[
        ScheduledTask[
            If[!$activeProblems["EmptyQ"], pushToPython[]], 
            0.008
        ]
    ];
);

waitForCompletion[] := (
    While[$activeRunningProblems["Length"] >= 1,
        Pause[0.1];
    ];
    If[$pipelineTask =!= None,
        TaskRemove[$pipelineTask];
        $pipelineTask = None;
    ];
    Print["[SYSTEM] Alle Aufgaben abgeschlossen. Pipeline-Task gestoppt."];
);

writeJSONL[state_Association, stream_] := 
    WriteLine[stream, ExportString[state /. None -> Null, "JSON", "Compact" -> True]];

writeErrorLog[data_, stream_] := If[stream =!= None, 
    WriteLine[stream, ExportString[data /. None -> Null, "JSON", "Compact" -> True]]
];

setUpExperiment[batchSize_] := Do[addFromProblemsPool[], {i, 1, $batchSize}]

addFromProblemsPool[] := With[{data = ProblemProvider`FetchNext[]},
	If[data === $Failed, 
		Print["Queue empty!"],
		data // pushtoactiveProblems // registertoHashTable]]



registertoHashTable[data_Association] := (
		$activeRunningProblems["Insert", data["uuid"]]
);


pullfromactiveProblems[] := If[$activeProblems["EmptyQ"], $Failed, $activeProblems["Pop"]];
pullfromrewardQueue[] := If[$rewardQueue["EmptyQ"], $Failed, $rewardQueue["Pop"]];


(** Pipeline Push Functions **)
handleState[state_Association] := 

    If[FailureQ[state],
        writeErrorLog[state, $failedLogStream]
    ,
        If[state["status"] == "reward_calc",
            handleRewardCalc[state],
            
            If[state["status"] == "training", 
                pushtoactiveProblems[state]
            ]
        ]
    ];

handleRewardCalc[state_Association]:= (
	If[state["inRewardCalc"] === 1, jobPush[$calculatedRewardPush, state],
	jobPush[$rewardPush, state]];
	pushToPython[];
)
    
pushtoactiveProblems[data_Association] := (
    If[$currentLogStream =!= None, writeJSONL[data, $currentLogStream]];
    $activeProblems["Push", data];
    data
);


pushToPython[] := Module[{prob}, 
	prob = pullfromactiveProblems[];
	If[prob =!= $Failed,
		jobPush[$nnProvider, prob]
	]
];

jobPush[socket_, state_Association] := Module[{jsonString, bytes}, 
    jsonString = ExportString[state, "JSON", "Compact" -> True];
    jsonString = StringDelete[jsonString, {"\n", "\r"}]; 
    bytes = StringToByteArray[jsonString, "UTF-8"];
    BinaryWrite[socket, bytes];
];

pushToSolver[state_Association] := Module[{solver},
	solver = state["solver"];
	If[solver == 0, jobPush[$newtonProvider, state], 
	jobPush[$gmgfProvider, state]
	]
]


(** Rewards **)
$rewardsQueue = CreateDataStructure["Queue"];
$rewardRegistry = CreateDataStructure["HashTable"];
$stateComparison = CreateDataStructure["HashTable"];

(*Step 1: Incoming reward_calc from solver*)

registerToRewardRegistry[rewardstate_Association, storedstate_Association] := (
	$rewardRegistry["Insert", rewardstate["uuid"] -> storedstate];
)

prepareRewardCalculation[state_Association] := Module[{newtonState, gmgfState, comparisonID}, 
	comparisonID = CreateUUID[];
	newtonState = setRewardState[state, 0, comparisonID];
	gmgfState = setRewardState[state, 1, comparisonID];
	registerToRewardRegistry[newtonState, state];
	registerToRewardRegistry[gmgfState, state];
	pushToSolver[newtonState];
	pushToSolver[gmgfState];
]

setRewardState[data_Association, control_Integer, comparisonID_String] := Module[{
	id = data["id"],
    yT = data["yTarget"],
    ep = data["epoch"],
    ns = 0,
    solver = control,
    x = data["initialx0"] 
	},
	Join[data, <|
        "status" -> "init",
        "initialx0" -> x,
        "uuid" -> CreateUUID[],
        "comparisonID" -> comparisonID,
        "epoch" -> ep,
        "inRewardCalc" -> 1,
        "networkStep" -> ns,
        "parentStateId" -> "initialState",
        "networkJobId" -> "init",
        "tolerance" -> 0.000000000000001,
        "maxIter" -> 1500,
        "stateId" -> TemplateApply["``_yTarget_``_epoch_``_networkstep_``", {id, yT, ep, ns}],
        "mask" -> 1,
        "solver" -> solver,
        "absTime" -> 0,
        "currentX" -> x,
        "iterSteps" -> 0,
        "localMaxTolerance" -> 0.0,
        "lastStepError" -> 0.0,
        "fx" -> x,
        "dfx" -> 0.0,
        "ddfx" -> 0.0,
        "kappa" -> 0,
        "lastKappa" -> 0
    |>]
]

(* Finished Workflow Step 1 *)

(* Step 2: Send reward calc step to solvers and wait for result *)

incomingStateComparison[state_Association] := Module[{compuuid},
    compuuid = state["comparisonID"];
    If[$stateComparison["KeyExistsQ", compuuid],
        absTimeComparison[state, compuuid] // finalizeTrainingStep,
        $stateComparison["Insert", compuuid -> state]
    ]
]

absTimeComparison[state_Association, comuuid_String] := Module[{compstate, stateabsTime, compabsTime, bestState}, 
	compstate = $stateComparison["Lookup", comuuid];
	stateabsTime = state["absTime"];
	compabsTime = compstate["absTime"];
	bestState = If[stateabsTime < compabsTime, state, compstate];
	$stateComparison["KeyDrop", comuuid];
	<|
        "BenchmarkabsTime" -> bestState["absTime"],
        "Benchmarksolver" -> bestState["solver"],
        "deleteState1" -> state["uuid"],
        "deleteState2" -> compstate["uuid"]
        |>
]

finalizeState[state_Association] := Module[{finalState, delete1, delete2, BenchmarkabsTime, Benchmarksolver, updatedState},
	delete1 = state["deleteState1"];
	delete2 = state["deleteState2"];
	finalState = $rewardRegistry["Lookup", delete1];
	$rewardRegistry["KeyDrop", delete1];
	$rewardRegistry["KeyDrop", delete2];
	BenchmarkabsTime = state["BenchmarkabsTime"];
	Benchmarksolver = state["Benchmarksolver"];
	updatedState = Join[finalState, <|
        "BenchmarkabsTime" -> BenchmarkabsTime,
        "Benchmarksolver" -> Benchmarksolver,
        "status" -> "finished"
    |>];
    If[$currentLogStream =!= None, writeJSONL[updatedState, $currentLogStream]];
    updatedState
]

resetToInit[data_Association] := Module[{
    id = data["id"],
    yT = data["yTarget"],
    ns = 0,
    x = data["initialx0"],
    yMin = data["yMin"],
    yMax = data["yMax"], 
    ep = data["epoch"],
    absTime = data["absTime"],
    recordAbsTime = data["recordAbsTime"], 
    record   
}, 
	If[absTime < recordAbsTime, record = absTime, record = recordAbsTime];
    Join[data, <|
        "status" -> "init",
        "initialx0" -> data["x0"],
        "uuid" -> TemplateApply["_id_``_yTarget_``_x0_", {id, yT, x}],
        "networkStep" -> ns,
        "parentStateId" -> "initialState",
        "rewardSolver" -> 0,
        "networkJobId" -> "init",
        "tolerance" -> 0.000000000000001,
        "maxIter" -> 1500,
        "inRewardCalc" -> 0,
        "yMin" -> yMin,
        "yMax" -> yMax,
        "stateId" -> TemplateApply["``_yTarget_``_epoch_``_networkstep_``", {id, yT, ep, ns}], 
        "mask" -> 1,
        "solver" -> 1,
        "absTime" -> 0,
        "recordAbsTime" -> record,
        "currentX" -> data["x0"],
        "iterSteps" -> 0,
        "localMaxTolerance" -> 0.0,
        "lastStepError" -> 0.0,
        "fx" -> data["x0"],
        "dfx" -> 0.0,
        "ddfx" -> 0.0,
        "kappa" -> 0,
        "lastKappa" -> 0
    |>]
]

bumpEpochandPush[state_Association] := Module[{currentEpoch, nextEpochState}, 
	
	currentEpoch = state["epoch"];
	currentEpoch++;
	$activeRunningProblems["Delete", state["uuid"]];
	addFromProblemsPool[];
	nextEpochState = Join[state, <|
		"epoch" -> currentEpoch
	|>];
	If[currentEpoch <= $maxEpoch, ProblemProvider`PushToDataQueue[nextEpochState], Print["[PIPELINE] Training Finished - not pushing new states!"]];
]

rewardPushPython[state_Association] := Module[{}, 
	jobPush[$resultsProvider, state];
	state
]

finalizeTrainingStep[state_] := (
	state //
	finalizeState //
	rewardPushPython //
	resetToInit //
	bumpEpochandPush
)


(*Communication*)
parseMessage[event_] := Module[{rawMsg, jsonString, jsonObject},
    rawMsg = event["DataByteArray"];
    jsonString = ByteArrayToString[rawMsg, "UTF-8"];
    jsonObject = Association @ ImportString[jsonString, "JSON"];
    Print["PARSE_MSG Received json Object: ", jsonObject];
    jsonObject   
]

internalparseMessage[event_] := Module[{rawMsg, jsonString, jsonObject},
    rawMsg = event["DataByteArray"];
    jsonString = ByteArrayToString[rawMsg, "UTF-8"];
    jsonObject = Association @ ImportString[jsonString, "JSON"];
    Print["Internal Received json Object: ", jsonObject];
    jsonObject   
]




$nnSink = SocketConnect[$nnSinkPORT, "ZMQ_PULL"]


nnSinkListener = SocketListen[$nnSink, parseMessage /* pushToSolver]


$newtonProvider = SocketOpen[$newtonTaskPORT, "ZMQ_PUSH"]
$newtonSink = SocketConnect[$newtonResPORT, "ZMQ_PULL"]
newtonSinkListener = SocketListen[$newtonSink, internalparseMessage /* handleState]


$gmgfProvider = SocketOpen[$gmgfTaskPORT, "ZMQ_PUSH"]
$gmgfSink = SocketConnect[$gmgfResPORT, "ZMQ_PULL"]
gmgfSinkListener = SocketListen[$gmgfSink, internalparseMessage /* handleState]


$rewardPush = SocketOpen[$rewardPushPORT, "ZMQ_PUSH"];
$rewardSink = SocketConnect[$rewardPushPORT, "ZMQ_PULL"]
$rewardSinkListener = SocketListen[$rewardSink, internalparseMessage /* prepareRewardCalculation]


StartPipeline[] := (
	Print["Provider Ready, waiting for connection!"];
	
	
	
	$solverProcess = StartProcess[{
    "wolframscript", "-file", "solverRunner.wls",
    "-newtonTaskPort", ToString[$newtonTaskPORT],
    "-newtonResPort",  ToString[$newtonResPORT],
    "-gmgfTaskPort", ToString[$gmgfTaskPORT],
	"-gmgfResPort", ToString[$gmgfResPORT]
}];
	$nnProvider = SocketConnect[$nnProviderPORT, "ZMQ_PUSH"];
	
	

	$calculatedRewardPush = SocketOpen[$calculatedRewardPushPORT, "ZMQ_PUSH"];
	$calculatedRewardSink = SocketConnect[$calculatedRewardPushPORT, "ZMQ_PULL"];
	$calculatedRewardListener = SocketListen[$calculatedRewardSink, internalparseMessage /* incomingStateComparison];
	$resultsProvider = SocketConnect[$resultsPORT, "ZMQ_PUSH"];
	
    Print["[NN-DAEMON] Ready!"];

	(*===== REWARD SOCKETS ====*)
	
	
	
	(*========================*)
	Print["[gMGF-DAEMON] Ready!"];
	Print["[NEWTON-DAEMON] Ready!"];
	setUpExperiment[$batchSize];
	pushToPython[];
	Print["[SYSTEM] Active Problems Initialised"];
	startPipelineTask[];
	While[TrueQ[$running],
    Pause[10];
	];
);


StartPipeline[];


Print["--- DAEMON OUTPUT ---"];
Print[ReadString[$solverProcess, EndOfBuffer]];

Print["--- DAEMON ERRORS ---"];
Print[ReadString[ProcessConnection[$solverProcess, "StandardError"], EndOfBuffer]];
