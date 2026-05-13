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

(*
  `$batchSize` only seeds `$activeProblems` at startup. A periodic pump pops
  one pending state per tick and pushes it to Python; there is no in-flight
  cap tied to batch size. Training returns and epoch rollovers refill the queue.
*)

$pythonDispatchInterval = 0.2;

startPipelineTask[] := (
    If[$pipelineTask =!= None, TaskRemove[$pipelineTask]];
    $pipelineTask = SessionSubmit[
        ScheduledTask[
            If[!$activeProblems["EmptyQ"], pushToPython[]],
            $pythonDispatchInterval
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
        Which[
            state["status"] == "error" || state["status"] == "non_converged",
                handleSolverError[state],
            state["status"] == "reward_calc",
                handleRewardCalc[state],
            state["status"] == "training",
                pushtoactiveProblems[state],
            True,
                writeErrorLog[<|"reason" -> "unknown_status", "state" -> state|>, $failedLogStream]
        ]
    ];

(*
  handleSolverError: a solver returned a state with status=error or non_converged.
  We log it, clean up registries so the orchestrator does not leak state,
  and (a) if it was a reward-calc leg, fail the comparison cleanly,
       (b) otherwise re-queue a fresh problem so the worker pool stays full.

*)
handleSolverError[state_Association] := Module[
    {uuid, compID, inReward, parentState, parentUuid},
    writeErrorLog[state, $failedLogStream];
    uuid     = Lookup[state, "uuid", None];
    compID   = Lookup[state, "comparisonID", None];
    inReward = TrueQ[Lookup[state, "inRewardCalc", 0] == 1];

    parentState = If[uuid =!= None && $rewardRegistry["KeyExistsQ", uuid],
        $rewardRegistry["Lookup", uuid]
        ,
        None
    ];

    If[uuid =!= None && $rewardRegistry["KeyExistsQ", uuid],
        $rewardRegistry["KeyDrop", uuid]
    ];

    If[inReward && compID =!= None,
        (*
          Reward-calc leg failed. The other leg may already have arrived
          (in $stateComparison) or may still be running. Either way, spoil
          the comparison: drop any pending partner, mark the ID so a future
          partner gets dropped on arrival, and free the parent's outer
          uuid in $activeRunningProblems so the training pool stays full.
        *)
        If[$stateComparison["KeyExistsQ", compID],
            With[{partner = $stateComparison["Lookup", compID]},
                If[$rewardRegistry["KeyExistsQ", partner["uuid"]],
                    $rewardRegistry["KeyDrop", partner["uuid"]]
                ];
                $stateComparison["KeyDrop", compID];
            ],
            $spoiledComparisons["Insert", compID]
        ];

        parentUuid = If[parentState =!= None, Lookup[parentState, "uuid", None], None];
        If[parentUuid =!= None,
            $activeRunningProblems["Delete", parentUuid];
            addFromProblemsPool[];
        ];
    ];

    If[!inReward && uuid =!= None,
        $activeRunningProblems["Delete", uuid];
        addFromProblemsPool[];
    ];
    state
];

handleRewardCalc[state_Association]:= (
	If[TrueQ[state["inRewardCalc"] == 1],
		jobPush[$calculatedRewardPush, state]
	,
		jobPush[$rewardPush, state]
	];
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

(*
  pushToSolver: route a state to either the Newton or gMGF solver.
  Solver==0 -> Newton, anything else -> gMGF. Uses numeric (==) comparison
  so a JSON-round-tripped Real (1.0) is treated the same as Integer (1).
*)
pushToSolver[state_Association] := Module[{solver},
	solver = Lookup[state, "solver", 1];
	If[TrueQ[solver == 0],
		jobPush[$newtonProvider, state],
		jobPush[$gmgfProvider, state]
	]
]


(** Rewards **)
$rewardsQueue = CreateDataStructure["Queue"];
$rewardRegistry = CreateDataStructure["HashTable"];
$stateComparison = CreateDataStructure["HashTable"];
(*
  $spoiledComparisons tracks comparisonIDs whose first leg failed in the
  solver. When the surviving leg arrives we know to discard it cleanly
  rather than wait forever for a partner that will never come.
*)
$spoiledComparisons = CreateDataStructure["HashSet"];

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

(*
  incomingStateComparison: pair up the two reward-calc legs by comparisonID.

  If the comparison is already spoiled (the other leg errored in the solver),
  drop this surviving leg cleanly. Otherwise insert into pending or finalize.
*)
incomingStateComparison[state_Association] := Module[{compuuid, uuid},
    compuuid = state["comparisonID"];
    uuid = state["uuid"];
    If[$spoiledComparisons["MemberQ", compuuid],
        $spoiledComparisons["Remove", compuuid];
        If[$rewardRegistry["KeyExistsQ", uuid],
            $rewardRegistry["KeyDrop", uuid]
        ];
        Return[Null]
    ];
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

(*
  resetToInit: prepare a finished state for the next epoch.

  IMPORTANT: A fresh ``uuid`` (CreateUUID[]) is generated per epoch so that
  the Python replay buffer (keyed by uuid) never overwrites a still-in-flight
  episode of the same (id, yTarget). The old deterministic template UUID
  caused silent episode-buffer corruption whenever the new epoch state hit
  Python while a prior epoch was still mid-step.
*)
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
        "uuid" -> CreateUUID[],
        "networkStep" -> ns,
        "parentStateId" -> "initialState",
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
	nextEpochState = Join[state, <|"epoch" -> currentEpoch|>];
	If[currentEpoch <= $maxEpoch,
		state // pushtoactiveProblems // registertoHashTable;
		ProblemProvider`PushToDataQueue[nextEpochState],
		Print["[PIPELINE] Training Finished - not pushing new states!"]
	];
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
(*
  parseMessage / internalparseMessage: ZMQ event -> Association.
  Print[] calls were removed from the hot path; failures surface via
  the unknown-status branch of handleState which logs to $failedLogStream.
*)
parseMessage[event_] := Module[{rawMsg, jsonString, jsonObject},
    rawMsg = event["DataByteArray"];
    jsonString = ByteArrayToString[rawMsg, "UTF-8"];
    jsonObject = Quiet @ Check[Association @ ImportString[jsonString, "JSON"], $Failed];
    If[jsonObject === $Failed,
        Return[<|"status" -> "error", "errorMessage" -> "Invalid JSON from Python"|>]
    ];
    jsonObject
]

internalparseMessage[event_] := Module[{rawMsg, jsonString, jsonObject},
    rawMsg = event["DataByteArray"];
    jsonString = ByteArrayToString[rawMsg, "UTF-8"];
    jsonObject = Quiet @ Check[Association @ ImportString[jsonString, "JSON"], $Failed];
    If[jsonObject === $Failed,
        Return[<|"status" -> "error", "errorMessage" -> "Invalid JSON from solver"|>]
    ];
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
	Print["[SYSTEM] Active Problems Initialised"];
	startPipelineTask[];
);


StartPipeline[];
waitForCompletion[];


Print["--- DAEMON OUTPUT ---"];
Print[ReadString[$solverProcess, EndOfBuffer]];

Print["--- DAEMON ERRORS ---"];
Print[ReadString[ProcessConnection[$solverProcess, "StandardError"], EndOfBuffer]];
