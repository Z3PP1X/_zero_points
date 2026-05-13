python3 main.py --experiment nur_f --timesteps 10000 --n_trials 10 --n_envs 32
Starte Pipeline mit Graphen aus: graphs/nur_f
[Gateway Debug] Loop is live! Polling on ports 5650 and 5693...
[I 2026-05-13 10:40:41,515] Using an existing study with name 'gnn_rl_nur_f' instead of creating a new one.

--- Starting Trial 19 ---
Nachricht uuid=e2ca411c-8649-447c-8293-f5ab9188d6d7 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=57ca5a90-0faa-4cc6-bb3a-1cb8a1d3031a passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=f700bcae-b46d-4c86-a783-20455dde1743 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=02823f67-5c4a-43d4-9a72-3052d7208fd5 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=7c8ee561-9c77-4ca6-aceb-f25fbc9ddda4 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=af5c96b2-1e52-46bb-881e-f07e362cf013 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=dcdd98e0-613f-418e-8f11-f33906770649 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=d6dfa67e-bf8d-4813-9f60-1a4ff7617cca passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=068a00de-9529-4ffc-b892-c3d7fab1675e passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=fc7828bb-8d2c-4256-8d06-fa41f65f6516 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=bc776fa8-3d62-4a2a-98c1-acc1120b3c6d passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=5c48b670-75e6-4086-a9a9-b1aae51c74ef passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=3dedb242-a2cc-4c6e-acfc-b9a816a1c243 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=1c6f78f6-1ee6-431d-a13e-09a6b8549516 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=1da1ce69-7367-414e-b4b2-348eb68c2cae passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=696852ae-d123-407b-bd80-bf186f209def passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=8ea7878d-9bd3-4229-9da2-d672c11eb282 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=45afdd6e-88d9-4700-80be-f3fb61b86489 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=c2692e80-e7ea-4808-9575-cb86e96c7931 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=97c0584d-1b34-4480-88b7-5ceddfa29c91 passt zu keinem erwarteten Slot — verworfen.
Nachricht uuid=b34331c1-2e75-4016-97e0-f9a990bd4cef passt zu keinem erwarteten Slot — verworfen.
step_wait Timeout nach 30.0s — Slots ohne Antwort: [0, 2, 5, 6, 21, 24, 26, 30] (uuids=['bb7fe8ac-e39e-4b59-8bf6-98c353d36d53', 'b6e9d354-36c0-4e35-9972-6626d8eb480c', '2c2a740f-0631-4173-b04d-f88299b8be2d', '75e3f2f9-cddc-43fa-8835-eedb8dc0fa8a', 'acce2030-6664-4a2d-83d4-0be12edd7b5f', '638f52cd-ef20-439a-a240-1d364b3f3b25', 'bbabb0b3-d6a8-4e6d-9e20-f9a283c6c2c1', '5ae5fbae-b97a-4de3-9828-653d322df76a']).



======

Mathematica: 

In[92]:= $nnSink = SocketConnect[$nnSinkPORT, "ZMQ_PULL"]

Out[92]= SocketObject["ae76a9b6-e5d5-452d-958d-988aa44ff3ba"]

In[93]:= nnSinkListener = SocketListen[$nnSink, parseMessage /* pushToSolver]

Out[93]= SocketListener[3819646809623339656]

In[94]:= $newtonProvider = SocketOpen[$newtonTaskPORT, "ZMQ_PUSH"]
$newtonSink = SocketConnect[$newtonResPORT, "ZMQ_PULL"]
newtonSinkListener = SocketListen[$newtonSink, internalparseMessage /* handleState]

Out[94]= SocketObject["42fd8bb6-0dfa-49c6-a7cf-58ebc67db3b0"]

Out[95]= SocketObject["724b948a-75e1-4d21-bfb2-29fe5ae4995c"]

Out[96]= SocketListener[3819646811822949519]

In[97]:= $gmgfProvider = SocketOpen[$gmgfTaskPORT, "ZMQ_PUSH"]
$gmgfSink = SocketConnect[$gmgfResPORT, "ZMQ_PULL"]
gmgfSinkListener = SocketListen[$gmgfSink, internalparseMessage /* handleState]

Out[97]= SocketObject["f3150ca7-9592-4e6e-8612-b64928b47618"]

Out[98]= SocketObject["e0a297a5-d064-4944-947c-e515a54e14c0"]

Out[99]= SocketListener[3819646812460966175]

In[100]:= $rewardPush = SocketOpen[$rewardPushPORT, "ZMQ_PUSH"];
$rewardSink = SocketConnect[$rewardPushPORT, "ZMQ_PULL"]
$rewardSinkListener = SocketListen[$rewardSink, internalparseMessage /* prepareRewardCalculation]

Out[101]= SocketObject["368fbdce-6313-43ea-8541-fb85dcb08b90"]

Out[102]= SocketListener[3819646811203129008]

Daemon: 

Print["--- DAEMON OUTPUT ---"];
Print[ReadString[$solverProcess, EndOfBuffer]];

Print["--- DAEMON ERRORS ---"];
Print[ReadString[ProcessConnection[$solverProcess, "StandardError"], EndOfBuffer]];

--- DAEMON OUTPUT ---

[NEWTON-DAEMON] Sockets connected, listening for Events...

[NEWTON-DAEMON] Starte Newton-Solver fr State ID: P1

[NEWTON-DAEMON] Starte Newton-Solver fr State ID: P6

[NEWTON-DAEMON] Starte Newton-Solver fr State ID: P1

[NEWTON-DAEMON] Starte Newton-Solver fr State ID: P7



FindRoot::nlnum: The function value {-0.1 + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.



FindRoot::nlnum: The function value {-99. + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.



FindRoot::nlnum: The function value {-0.1 + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.



General::stop: Further output of FindRoot::nlnum will be suppressed during this calculation.



Part::partw: Part 1 of {} does not exist.



Part::partw: Part 1 of {} does not exist.



Part::partw: Part 1 of {} does not exist.



General::stop: Further output of Part::partw will be suppressed during this calculation.



[\|01f6a8 NEWTON SOLVER CRASH / WARNUNG \|01f6a8]

FindRoot::nlnum : The function value {-0.1 + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

FindRoot::nlnum : The function value {-99. + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

FindRoot::nlnum : The function value {-0.1 + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

General::stop : Further output of FindRoot::nlnum will be suppressed during this calculation.

FindRoot::nlnum : The function value {-99. + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

Part::partw : Part 1 of {} does not exist.

Part::partw : Part 1 of {} does not exist.

Part::partw : Part 1 of {} does not exist.

General::stop : Further output of Part::partw will be suppressed during this calculation.

[\|01f6a8 ============================== \|01f6a8]



[NEWTON-DAEMON] Starte Newton-Solver fr State ID: P7



[\|01f6a8 NEWTON SOLVER CRASH / WARNUNG \|01f6a8]

FindRoot::nlnum : The function value {-0.1 + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

FindRoot::nlnum : The function value {-99. + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

FindRoot::nlnum : The function value {-0.1 + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

FindRoot::nlnum : The function value {-99. + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

Part::partw : Part 1 of {} does not exist.

Part::partw : Part 1 of {} does not exist.

Part::partw : Part 1 of {} does not exist.

[\|01f6a8 ============================== \|01f6a8]



[NEWTON-DAEMON] Starte Newton-Solver fr State ID: P7



[\|01f6a8 NEWTON SOLVER CRASH / WARNUNG \|01f6a8]

FindRoot::nlnum : The function value {-0.1 + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

FindRoot::nlnum : The function value {-99. + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

FindRoot::nlnum : The function value {-0.1 + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

FindRoot::nlnum : The function value {-99. + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {newtonSolver`Private`x} = {1.}.

Part::partw : Part 1 of {} does not exist.

Part::partw : Part 1 of {} does not exist.

Part::partw : Part 1 of {} does not exist.

[\|01f6a8 ============================== \|01f6a8]



[NEWTON-DAEMON] Starte Newton-Solver fr State ID: P6

[NEWTON-DAEMON] Starte Newton-Solver fr State ID: P6

[NEWTON-DAEMON] Starte Newton-Solver fr State ID: P5

[NEWTON-DAEMON] Starte Newton-Solver fr State ID: P1

[NEWTON-DAEMON] Starte Newton-Solver fr State ID: P1

[GMGF-DAEMON] Sockets connected, listening for Events...

[GMGF-DAEMON] Starte gMGF-Solver fr State ID: P6

[GMGF-DAEMON] Starte gMGF-Solver fr State ID: P7



FindRoot::nlnum: The function value {-0.1 + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {gmgfSolver`Private`x} = {1.}.



FindRoot::nlnum: The function value {-99. + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {gmgfSolver`Private`x} = {1.}.



FindRoot::nlnum: The function value {-0.1 + Missing[KeyAbsent, P7][1.]} is not a list of numbers with dimensions {1} at {gmgfSolver`Private`x} = {1.}.



General::stop: Further output of FindRoot::nlnum will be suppressed during this calculation.



Part::partw: Part 1 of {} does not exist.



Part::partw: Part 1 of {} does not exist.



Part::partw: Part 1 of {} does not exist.



General::stop: Further output of Part::partw will be suppressed during this calculation.



                                                                                                 Sign[-83.5562 + Missing[KeyAbsent, P7][1.]] {}[[1]][Abs[-83.5562 + Missing[KeyAbsent, P7][1.]]]         -15                                                                                                                                                                                                                                                                                                                                      -15                                                                                                                                                                                         9                   Sign[-83.5562 + Missing[KeyAbsent, P7][1.]] {}[[1]][Abs[-83.5562 + Missing[KeyAbsent, P7][1.]]]                                            -11                            Sign[-83.5562 + Missing[KeyAbsent, P7][1.]] {}[[1]][Abs[-83.5562 + Missing[KeyAbsent, P7][1.]]]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            -15                                                                                                                                                                                         9                   Sign[-83.5562 + Missing[KeyAbsent, P7][1.]] {}[[1]][Abs[-83.5562 + Missing[KeyAbsent, P7][1.]]]                                            -11                            Sign[-83.5562 + Missing[KeyAbsent, P7][1.]] {}[[1]][Abs[-83.5562 + Missing[KeyAbsent, P7][1.]]]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  Sign[-83.5562 + Missing[KeyAbsent, P7][1.]] {}[[1]][Abs[-83.5562 + Missing[KeyAbsent, P7][1.]]]         -15                                                                                                                                                                                                                                                                                                                                      -15                                                                                                                                                                                         9                   Sign[-83.5562 + Missing[KeyAbsent, P7][1.]] {}[[1]][Abs[-83.5562 + Missing[KeyAbsent, P7][1.]]]                                            -11                            Sign[-83.5562 + Missing[KeyAbsent, P7][1.]] {}[[1]][Abs[-83.5562 + Missing[KeyAbsent, P7][1.]]]      

--- DAEMON ERRORS ---

