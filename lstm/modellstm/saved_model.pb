±є+
≥Г
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resourceИ
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
Ѓ
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
≥
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
П
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeКнout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
Ъ
StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
∞
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
Я
TensorListReserve
element_shape"
shape_type
num_elements(
handleКйиelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint€€€€€€€€€
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.13.02v2.13.0-rc2-7-g1cb1a030a628пч)
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
А
Adam/v/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_10/bias
y
(Adam/v/dense_10/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_10/bias*
_output_shapes
:*
dtype0
А
Adam/m/dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_10/bias
y
(Adam/m/dense_10/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_10/bias*
_output_shapes
:*
dtype0
И
Adam/v/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/v/dense_10/kernel
Б
*Adam/v/dense_10/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_10/kernel*
_output_shapes

:*
dtype0
И
Adam/m/dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/m/dense_10/kernel
Б
*Adam/m/dense_10/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_10/kernel*
_output_shapes

:*
dtype0
~
Adam/v/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_9/bias
w
'Adam/v/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_9/bias
w
'Adam/m/dense_9/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/bias*
_output_shapes
:*
dtype0
Ж
Adam/v/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/v/dense_9/kernel

)Adam/v/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_9/kernel*
_output_shapes

:*
dtype0
Ж
Adam/m/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/m/dense_9/kernel

)Adam/m/dense_9/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_9/kernel*
_output_shapes

:*
dtype0
~
Adam/v/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_8/bias
w
'Adam/v/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_8/bias
w
'Adam/m/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/bias*
_output_shapes
:*
dtype0
Ж
Adam/v/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/v/dense_8/kernel

)Adam/v/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/kernel*
_output_shapes

: *
dtype0
Ж
Adam/m/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/m/dense_8/kernel

)Adam/m/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/kernel*
_output_shapes

: *
dtype0
Н
Adam/v/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameAdam/v/lstm/lstm_cell/bias
Ж
.Adam/v/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/v/lstm/lstm_cell/bias*
_output_shapes	
:А*
dtype0
Н
Adam/m/lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_nameAdam/m/lstm/lstm_cell/bias
Ж
.Adam/m/lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOpAdam/m/lstm/lstm_cell/bias*
_output_shapes	
:А*
dtype0
©
&Adam/v/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*7
shared_name(&Adam/v/lstm/lstm_cell/recurrent_kernel
Ґ
:Adam/v/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp&Adam/v/lstm/lstm_cell/recurrent_kernel*
_output_shapes
:	 А*
dtype0
©
&Adam/m/lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*7
shared_name(&Adam/m/lstm/lstm_cell/recurrent_kernel
Ґ
:Adam/m/lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp&Adam/m/lstm/lstm_cell/recurrent_kernel*
_output_shapes
:	 А*
dtype0
Х
Adam/v/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
А*-
shared_nameAdam/v/lstm/lstm_cell/kernel
О
0Adam/v/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/v/lstm/lstm_cell/kernel*
_output_shapes
:	
А*
dtype0
Х
Adam/m/lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
А*-
shared_nameAdam/m/lstm/lstm_cell/kernel
О
0Adam/m/lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpAdam/m/lstm/lstm_cell/kernel*
_output_shapes
:	
А*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	

lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*$
shared_namelstm/lstm_cell/bias
x
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes	
:А*
dtype0
Ы
lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 А*0
shared_name!lstm/lstm_cell/recurrent_kernel
Ф
3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel*
_output_shapes
:	 А*
dtype0
З
lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
А*&
shared_namelstm/lstm_cell/kernel
А
)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel*
_output_shapes
:	
А*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

: *
dtype0
Ч
serving_default_lstm_inputPlaceholder*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
*
dtype0*)
shape :€€€€€€€€€€€€€€€€€€

ю
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_inputlstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biasdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_1310977

NoOpNoOp
к=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*•=
valueЫ=BШ= BС=
и
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
¶
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¶
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
¶
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
C
/0
01
12
3
4
%5
&6
-7
.8*
C
/0
01
12
3
4
%5
&6
-7
.8*
* 
∞
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*

7trace_0
8trace_1* 

9trace_0
:trace_1* 
* 
Б
;
_variables
<_iterations
=_learning_rate
>_index_dict
?
_momentums
@_velocities
A_update_step_xla*

Bserving_default* 

/0
01
12*

/0
01
12*
* 
Я

Cstates
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_3* 
6
Mtrace_0
Ntrace_1
Otrace_2
Ptrace_3* 
* 
г
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator
X
state_size

/kernel
0recurrent_kernel
1bias*
* 

0
1*

0
1*
* 
У
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

^trace_0* 

_trace_0* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*
* 
У
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

etrace_0* 

ftrace_0* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 
У
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

ltrace_0* 

mtrace_0* 
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUElstm/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUElstm/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

n0
o1*
* 
* 
* 
* 
* 
* 
Ф
<0
p1
q2
r3
s4
t5
u6
v7
w8
x9
y10
z11
{12
|13
}14
~15
16
А17
Б18*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
D
p0
r1
t2
v3
x4
z5
|6
~7
А8*
D
q0
s1
u2
w3
y4
{5
}6
7
Б8*
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

/0
01
12*

/0
01
12*
* 
Ш
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
З	variables
И	keras_api

Йtotal

Кcount*
M
Л	variables
М	keras_api

Нtotal

Оcount
П
_fn_kwargs*
ga
VARIABLE_VALUEAdam/m/lstm/lstm_cell/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/lstm/lstm_cell/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/m/lstm/lstm_cell/recurrent_kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&Adam/v/lstm/lstm_cell/recurrent_kernel1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/m/lstm/lstm_cell/bias1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEAdam/v/lstm/lstm_cell/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_8/kernel1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_8/kernel1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_8/bias1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_8/bias2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/dense_9/kernel2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_9/kernel2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_9/bias2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_9/bias2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_10/kernel2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_10/kernel2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_10/bias2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_10/bias2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 

Й0
К1*

З	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

Н0
О1*

Л	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
р
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/bias	iterationlearning_rateAdam/m/lstm/lstm_cell/kernelAdam/v/lstm/lstm_cell/kernel&Adam/m/lstm/lstm_cell/recurrent_kernel&Adam/v/lstm/lstm_cell/recurrent_kernelAdam/m/lstm/lstm_cell/biasAdam/v/lstm/lstm_cell/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biasAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/biasAdam/m/dense_10/kernelAdam/v/dense_10/kernelAdam/m/dense_10/biasAdam/v/dense_10/biastotal_1count_1totalcountConst*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_save_1313069
л
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/biaslstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/bias	iterationlearning_rateAdam/m/lstm/lstm_cell/kernelAdam/v/lstm/lstm_cell/kernel&Adam/m/lstm/lstm_cell/recurrent_kernel&Adam/v/lstm/lstm_cell/recurrent_kernelAdam/m/lstm/lstm_cell/biasAdam/v/lstm/lstm_cell/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biasAdam/m/dense_9/kernelAdam/v/dense_9/kernelAdam/m/dense_9/biasAdam/v/dense_9/biasAdam/m/dense_10/kernelAdam/v/dense_10/kernelAdam/m/dense_10/biasAdam/v/dense_10/biastotal_1count_1totalcount*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__traced_restore_1313177Чг(
°

∆
while_cond_1309476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1309476___redundant_placeholder05
1while_while_cond_1309476___redundant_placeholder15
1while_while_cond_1309476___redundant_placeholder25
1while_while_cond_1309476___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
™Ћ
х
=__inference___backward_gpu_lstm_with_fallback_1309657_1309833
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:€€€€€€€€€ m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:€€€€€€€€€ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:€€€€€€€€€ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: Л
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕЖ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:™
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
shrink_axis_maskЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:ј
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ§
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Г
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ®
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ В
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Й
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:ў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
Г
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::нѕ≈
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::нѕ…
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Аi
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ш
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::м
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ап
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: п
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   °
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:Ј
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:Ј
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:Ј
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:Ј
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:Ј
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:Ј
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:Ј
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  з
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Аѓ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
Аµ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::“
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А÷
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesс
о:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : :€€€€€€€€€€€€€€€€€€ ::€€€€€€€€€ :€€€€€€€€€ ::€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,::€€€€€€€€€ :€€€€€€€€€ : ::::::::: : : : *=
api_implements+)lstm_e35cb2d6-fa30-4ae4-ab3c-6ad53fcb0768*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1309832*
go_backwards( *

time_major( :C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:EA

_output_shapes	
:А,
"
_user_specified_name
concat_1:YU
+
_output_shapes
:€€€€€€€€€ 
&
_user_specified_nameExpandDims_1:WS
+
_output_shapes
:€€€€€€€€€ 
$
_user_specified_name
ExpandDims:_
[
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

#
_user_specified_name	transpose:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_9/perm:^Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ ::6
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ :- )
'
_output_shapes
:€€€€€€€€€ 
™Ћ
х
=__inference___backward_gpu_lstm_with_fallback_1309230_1309406
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:€€€€€€€€€ m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:€€€€€€€€€ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:€€€€€€€€€ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: Л
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕЖ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:™
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
shrink_axis_maskЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:ј
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ§
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Г
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ®
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ В
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Й
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:ў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
Г
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::нѕ≈
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::нѕ…
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Аi
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ш
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::м
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ап
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: п
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   °
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:Ј
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:Ј
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:Ј
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:Ј
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:Ј
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:Ј
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:Ј
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  з
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Аѓ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
Аµ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::“
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А÷
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesс
о:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : :€€€€€€€€€€€€€€€€€€ ::€€€€€€€€€ :€€€€€€€€€ ::€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,::€€€€€€€€€ :€€€€€€€€€ : ::::::::: : : : *=
api_implements+)lstm_d3eb9e0d-7c67-4700-a6f4-7ba2db9f8384*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1309405*
go_backwards( *

time_major( :C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:EA

_output_shapes	
:А,
"
_user_specified_name
concat_1:YU
+
_output_shapes
:€€€€€€€€€ 
&
_user_specified_nameExpandDims_1:WS
+
_output_shapes
:€€€€€€€€€ 
$
_user_specified_name
ExpandDims:_
[
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

#
_user_specified_name	transpose:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_9/perm:^Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ ::6
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ :- )
'
_output_shapes
:€€€€€€€€€ 
ою
І
 __inference__traced_save_1313069
file_prefix7
%read_disablecopyonread_dense_8_kernel: 3
%read_1_disablecopyonread_dense_8_bias:9
'read_2_disablecopyonread_dense_9_kernel:3
%read_3_disablecopyonread_dense_9_bias::
(read_4_disablecopyonread_dense_10_kernel:4
&read_5_disablecopyonread_dense_10_bias:A
.read_6_disablecopyonread_lstm_lstm_cell_kernel:	
АK
8read_7_disablecopyonread_lstm_lstm_cell_recurrent_kernel:	 А;
,read_8_disablecopyonread_lstm_lstm_cell_bias:	А,
"read_9_disablecopyonread_iteration:	 1
'read_10_disablecopyonread_learning_rate: I
6read_11_disablecopyonread_adam_m_lstm_lstm_cell_kernel:	
АI
6read_12_disablecopyonread_adam_v_lstm_lstm_cell_kernel:	
АS
@read_13_disablecopyonread_adam_m_lstm_lstm_cell_recurrent_kernel:	 АS
@read_14_disablecopyonread_adam_v_lstm_lstm_cell_recurrent_kernel:	 АC
4read_15_disablecopyonread_adam_m_lstm_lstm_cell_bias:	АC
4read_16_disablecopyonread_adam_v_lstm_lstm_cell_bias:	АA
/read_17_disablecopyonread_adam_m_dense_8_kernel: A
/read_18_disablecopyonread_adam_v_dense_8_kernel: ;
-read_19_disablecopyonread_adam_m_dense_8_bias:;
-read_20_disablecopyonread_adam_v_dense_8_bias:A
/read_21_disablecopyonread_adam_m_dense_9_kernel:A
/read_22_disablecopyonread_adam_v_dense_9_kernel:;
-read_23_disablecopyonread_adam_m_dense_9_bias:;
-read_24_disablecopyonread_adam_v_dense_9_bias:B
0read_25_disablecopyonread_adam_m_dense_10_kernel:B
0read_26_disablecopyonread_adam_v_dense_10_kernel:<
.read_27_disablecopyonread_adam_m_dense_10_bias:<
.read_28_disablecopyonread_adam_v_dense_10_bias:+
!read_29_disablecopyonread_total_1: +
!read_30_disablecopyonread_count_1: )
read_31_disablecopyonread_total: )
read_32_disablecopyonread_count: 
savev2_const
identity_67ИҐMergeV2CheckpointsҐRead/DisableCopyOnReadҐRead/ReadVariableOpҐRead_1/DisableCopyOnReadҐRead_1/ReadVariableOpҐRead_10/DisableCopyOnReadҐRead_10/ReadVariableOpҐRead_11/DisableCopyOnReadҐRead_11/ReadVariableOpҐRead_12/DisableCopyOnReadҐRead_12/ReadVariableOpҐRead_13/DisableCopyOnReadҐRead_13/ReadVariableOpҐRead_14/DisableCopyOnReadҐRead_14/ReadVariableOpҐRead_15/DisableCopyOnReadҐRead_15/ReadVariableOpҐRead_16/DisableCopyOnReadҐRead_16/ReadVariableOpҐRead_17/DisableCopyOnReadҐRead_17/ReadVariableOpҐRead_18/DisableCopyOnReadҐRead_18/ReadVariableOpҐRead_19/DisableCopyOnReadҐRead_19/ReadVariableOpҐRead_2/DisableCopyOnReadҐRead_2/ReadVariableOpҐRead_20/DisableCopyOnReadҐRead_20/ReadVariableOpҐRead_21/DisableCopyOnReadҐRead_21/ReadVariableOpҐRead_22/DisableCopyOnReadҐRead_22/ReadVariableOpҐRead_23/DisableCopyOnReadҐRead_23/ReadVariableOpҐRead_24/DisableCopyOnReadҐRead_24/ReadVariableOpҐRead_25/DisableCopyOnReadҐRead_25/ReadVariableOpҐRead_26/DisableCopyOnReadҐRead_26/ReadVariableOpҐRead_27/DisableCopyOnReadҐRead_27/ReadVariableOpҐRead_28/DisableCopyOnReadҐRead_28/ReadVariableOpҐRead_29/DisableCopyOnReadҐRead_29/ReadVariableOpҐRead_3/DisableCopyOnReadҐRead_3/ReadVariableOpҐRead_30/DisableCopyOnReadҐRead_30/ReadVariableOpҐRead_31/DisableCopyOnReadҐRead_31/ReadVariableOpҐRead_32/DisableCopyOnReadҐRead_32/ReadVariableOpҐRead_4/DisableCopyOnReadҐRead_4/ReadVariableOpҐRead_5/DisableCopyOnReadҐRead_5/ReadVariableOpҐRead_6/DisableCopyOnReadҐRead_6/ReadVariableOpҐRead_7/DisableCopyOnReadҐRead_7/ReadVariableOpҐRead_8/DisableCopyOnReadҐRead_8/ReadVariableOpҐRead_9/DisableCopyOnReadҐRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: w
Read/DisableCopyOnReadDisableCopyOnRead%read_disablecopyonread_dense_8_kernel"/device:CPU:0*
_output_shapes
 °
Read/ReadVariableOpReadVariableOp%read_disablecopyonread_dense_8_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

: y
Read_1/DisableCopyOnReadDisableCopyOnRead%read_1_disablecopyonread_dense_8_bias"/device:CPU:0*
_output_shapes
 °
Read_1/ReadVariableOpReadVariableOp%read_1_disablecopyonread_dense_8_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_9_kernel"/device:CPU:0*
_output_shapes
 І
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_9_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_9_bias"/device:CPU:0*
_output_shapes
 °
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_9_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:|
Read_4/DisableCopyOnReadDisableCopyOnRead(read_4_disablecopyonread_dense_10_kernel"/device:CPU:0*
_output_shapes
 ®
Read_4/ReadVariableOpReadVariableOp(read_4_disablecopyonread_dense_10_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:z
Read_5/DisableCopyOnReadDisableCopyOnRead&read_5_disablecopyonread_dense_10_bias"/device:CPU:0*
_output_shapes
 Ґ
Read_5/ReadVariableOpReadVariableOp&read_5_disablecopyonread_dense_10_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_6/DisableCopyOnReadDisableCopyOnRead.read_6_disablecopyonread_lstm_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ѓ
Read_6/ReadVariableOpReadVariableOp.read_6_disablecopyonread_lstm_lstm_cell_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
А*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
Аf
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	
АМ
Read_7/DisableCopyOnReadDisableCopyOnRead8read_7_disablecopyonread_lstm_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 є
Read_7/ReadVariableOpReadVariableOp8read_7_disablecopyonread_lstm_lstm_cell_recurrent_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 А*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 Аf
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	 АА
Read_8/DisableCopyOnReadDisableCopyOnRead,read_8_disablecopyonread_lstm_lstm_cell_bias"/device:CPU:0*
_output_shapes
 ©
Read_8/ReadVariableOpReadVariableOp,read_8_disablecopyonread_lstm_lstm_cell_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:Аv
Read_9/DisableCopyOnReadDisableCopyOnRead"read_9_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Ъ
Read_9/ReadVariableOpReadVariableOp"read_9_disablecopyonread_iteration^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	f
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_10/DisableCopyOnReadDisableCopyOnRead'read_10_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 °
Read_10/ReadVariableOpReadVariableOp'read_10_disablecopyonread_learning_rate^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
: Л
Read_11/DisableCopyOnReadDisableCopyOnRead6read_11_disablecopyonread_adam_m_lstm_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 є
Read_11/ReadVariableOpReadVariableOp6read_11_disablecopyonread_adam_m_lstm_lstm_cell_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
А*
dtype0p
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
Аf
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:	
АЛ
Read_12/DisableCopyOnReadDisableCopyOnRead6read_12_disablecopyonread_adam_v_lstm_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 є
Read_12/ReadVariableOpReadVariableOp6read_12_disablecopyonread_adam_v_lstm_lstm_cell_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	
А*
dtype0p
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	
Аf
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:	
АХ
Read_13/DisableCopyOnReadDisableCopyOnRead@read_13_disablecopyonread_adam_m_lstm_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 √
Read_13/ReadVariableOpReadVariableOp@read_13_disablecopyonread_adam_m_lstm_lstm_cell_recurrent_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 А*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 Аf
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	 АХ
Read_14/DisableCopyOnReadDisableCopyOnRead@read_14_disablecopyonread_adam_v_lstm_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 √
Read_14/ReadVariableOpReadVariableOp@read_14_disablecopyonread_adam_v_lstm_lstm_cell_recurrent_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	 А*
dtype0p
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	 Аf
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:	 АЙ
Read_15/DisableCopyOnReadDisableCopyOnRead4read_15_disablecopyonread_adam_m_lstm_lstm_cell_bias"/device:CPU:0*
_output_shapes
 ≥
Read_15/ReadVariableOpReadVariableOp4read_15_disablecopyonread_adam_m_lstm_lstm_cell_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЙ
Read_16/DisableCopyOnReadDisableCopyOnRead4read_16_disablecopyonread_adam_v_lstm_lstm_cell_bias"/device:CPU:0*
_output_shapes
 ≥
Read_16/ReadVariableOpReadVariableOp4read_16_disablecopyonread_adam_v_lstm_lstm_cell_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:АД
Read_17/DisableCopyOnReadDisableCopyOnRead/read_17_disablecopyonread_adam_m_dense_8_kernel"/device:CPU:0*
_output_shapes
 ±
Read_17/ReadVariableOpReadVariableOp/read_17_disablecopyonread_adam_m_dense_8_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes

: Д
Read_18/DisableCopyOnReadDisableCopyOnRead/read_18_disablecopyonread_adam_v_dense_8_kernel"/device:CPU:0*
_output_shapes
 ±
Read_18/ReadVariableOpReadVariableOp/read_18_disablecopyonread_adam_v_dense_8_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

: В
Read_19/DisableCopyOnReadDisableCopyOnRead-read_19_disablecopyonread_adam_m_dense_8_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_19/ReadVariableOpReadVariableOp-read_19_disablecopyonread_adam_m_dense_8_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_20/DisableCopyOnReadDisableCopyOnRead-read_20_disablecopyonread_adam_v_dense_8_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_20/ReadVariableOpReadVariableOp-read_20_disablecopyonread_adam_v_dense_8_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:Д
Read_21/DisableCopyOnReadDisableCopyOnRead/read_21_disablecopyonread_adam_m_dense_9_kernel"/device:CPU:0*
_output_shapes
 ±
Read_21/ReadVariableOpReadVariableOp/read_21_disablecopyonread_adam_m_dense_9_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_22/DisableCopyOnReadDisableCopyOnRead/read_22_disablecopyonread_adam_v_dense_9_kernel"/device:CPU:0*
_output_shapes
 ±
Read_22/ReadVariableOpReadVariableOp/read_22_disablecopyonread_adam_v_dense_9_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:В
Read_23/DisableCopyOnReadDisableCopyOnRead-read_23_disablecopyonread_adam_m_dense_9_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_23/ReadVariableOpReadVariableOp-read_23_disablecopyonread_adam_m_dense_9_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:В
Read_24/DisableCopyOnReadDisableCopyOnRead-read_24_disablecopyonread_adam_v_dense_9_bias"/device:CPU:0*
_output_shapes
 Ђ
Read_24/ReadVariableOpReadVariableOp-read_24_disablecopyonread_adam_v_dense_9_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:Е
Read_25/DisableCopyOnReadDisableCopyOnRead0read_25_disablecopyonread_adam_m_dense_10_kernel"/device:CPU:0*
_output_shapes
 ≤
Read_25/ReadVariableOpReadVariableOp0read_25_disablecopyonread_adam_m_dense_10_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes

:Е
Read_26/DisableCopyOnReadDisableCopyOnRead0read_26_disablecopyonread_adam_v_dense_10_kernel"/device:CPU:0*
_output_shapes
 ≤
Read_26/ReadVariableOpReadVariableOp0read_26_disablecopyonread_adam_v_dense_10_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes

:Г
Read_27/DisableCopyOnReadDisableCopyOnRead.read_27_disablecopyonread_adam_m_dense_10_bias"/device:CPU:0*
_output_shapes
 ђ
Read_27/ReadVariableOpReadVariableOp.read_27_disablecopyonread_adam_m_dense_10_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:Г
Read_28/DisableCopyOnReadDisableCopyOnRead.read_28_disablecopyonread_adam_v_dense_10_bias"/device:CPU:0*
_output_shapes
 ђ
Read_28/ReadVariableOpReadVariableOp.read_28_disablecopyonread_adam_v_dense_10_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_29/DisableCopyOnReadDisableCopyOnRead!read_29_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_29/ReadVariableOpReadVariableOp!read_29_disablecopyonread_total_1^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_30/DisableCopyOnReadDisableCopyOnRead!read_30_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_30/ReadVariableOpReadVariableOp!read_30_disablecopyonread_count_1^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_31/DisableCopyOnReadDisableCopyOnReadread_31_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_31/ReadVariableOpReadVariableOpread_31_disablecopyonread_total^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_32/DisableCopyOnReadDisableCopyOnReadread_32_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Щ
Read_32/ReadVariableOpReadVariableOpread_32_disablecopyonread_count^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
: ™
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*”
value…B∆"B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH±
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B  
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *0
dtypes&
$2"	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_66Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_67IdentityIdentity_66:output:0^NoOp*
T0*
_output_shapes
: ф
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_67Identity_67:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:="9

_output_shapes
: 

_user_specified_nameConst:%!!

_user_specified_namecount:% !

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:40
.
_user_specified_nameAdam/v/dense_10/bias:40
.
_user_specified_nameAdam/m/dense_10/bias:62
0
_user_specified_nameAdam/v/dense_10/kernel:62
0
_user_specified_nameAdam/m/dense_10/kernel:3/
-
_user_specified_nameAdam/v/dense_9/bias:3/
-
_user_specified_nameAdam/m/dense_9/bias:51
/
_user_specified_nameAdam/v/dense_9/kernel:51
/
_user_specified_nameAdam/m/dense_9/kernel:3/
-
_user_specified_nameAdam/v/dense_8/bias:3/
-
_user_specified_nameAdam/m/dense_8/bias:51
/
_user_specified_nameAdam/v/dense_8/kernel:51
/
_user_specified_nameAdam/m/dense_8/kernel::6
4
_user_specified_nameAdam/v/lstm/lstm_cell/bias::6
4
_user_specified_nameAdam/m/lstm/lstm_cell/bias:FB
@
_user_specified_name(&Adam/v/lstm/lstm_cell/recurrent_kernel:FB
@
_user_specified_name(&Adam/m/lstm/lstm_cell/recurrent_kernel:<8
6
_user_specified_nameAdam/v/lstm/lstm_cell/kernel:<8
6
_user_specified_nameAdam/m/lstm/lstm_cell/kernel:-)
'
_user_specified_namelearning_rate:)
%
#
_user_specified_name	iteration:3	/
-
_user_specified_namelstm/lstm_cell/bias:?;
9
_user_specified_name!lstm/lstm_cell/recurrent_kernel:51
/
_user_specified_namelstm/lstm_cell/kernel:-)
'
_user_specified_namedense_10/bias:/+
)
_user_specified_namedense_10/kernel:,(
&
_user_specified_namedense_9/bias:.*
(
_user_specified_namedense_9/kernel:,(
&
_user_specified_namedense_8/bias:.*
(
_user_specified_namedense_8/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
°

∆
while_cond_1312370
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1312370___redundant_placeholder05
1while_while_cond_1312370___redundant_placeholder15
1while_while_cond_1312370___redundant_placeholder25
1while_while_cond_1312370___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ъ*
–
while_body_1310474
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€
*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:€€€€€€€€€АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :»
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Є
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ "*
while_biasadd_biaswhile_biasadd_bias_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А:A
=

_output_shapes	
:А

_user_specified_namebias:Q	M

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ґ
ы
D__inference_dense_8_layer_call_and_return_conditional_losses_1310324

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::нѕY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
 :
Ѕ
!__inference_standard_lstm_1309562

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:€€€€€€€€€А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:€€€€€€€€€Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:€€€€€€€€€АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ґ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А* 
_read_only_resource_inputs
 *
bodyR
while_body_1309477*
condR
while_cond_1309476*b
output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_e35cb2d6-fa30-4ae4-ab3c-6ad53fcb0768*
api_preferred_deviceCPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
а@
Ќ
*__inference_gpu_lstm_with_fallback_1309229

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:А,”
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_d3eb9e0d-7c67-4700-a6f4-7ba2db9f8384*
api_preferred_deviceGPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
°

∆
while_cond_1311943
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1311943___redundant_placeholder05
1while_while_cond_1311943___redundant_placeholder15
1while_while_cond_1311943___redundant_placeholder25
1while_while_cond_1311943___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ъ*
–
while_body_1311944
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€
*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:€€€€€€€€€АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :»
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Є
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ "*
while_biasadd_biaswhile_biasadd_bias_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А:A
=

_output_shapes	
:А

_user_specified_namebias:Q	M

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
£
ї
A__inference_lstm_layer_call_and_return_conditional_losses_1310832

inputs/
read_readvariableop_resource:	
А1
read_1_readvariableop_resource:	 А-
read_2_readvariableop_resource:	А

identity_3ИҐRead/ReadVariableOpҐRead_1/ReadVariableOpҐRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:АЉ
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference_standard_lstm_1310559v

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ h
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
°

∆
while_cond_1309049
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1309049___redundant_placeholder05
1while_while_cond_1309049___redundant_placeholder15
1while_while_cond_1309049___redundant_placeholder25
1while_while_cond_1309049___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
™Ћ
х
=__inference___backward_gpu_lstm_with_fallback_1308722_1308898
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:€€€€€€€€€ m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:€€€€€€€€€ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:€€€€€€€€€ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: Л
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕЖ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:™
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
shrink_axis_maskЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:ј
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ§
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Г
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ®
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ В
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Й
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:ў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
Г
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::нѕ≈
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::нѕ…
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Аi
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ш
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::м
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ап
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: п
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   °
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:Ј
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:Ј
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:Ј
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:Ј
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:Ј
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:Ј
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:Ј
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  з
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Аѓ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
Аµ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::“
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А÷
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesс
о:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : :€€€€€€€€€€€€€€€€€€ ::€€€€€€€€€ :€€€€€€€€€ ::€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,::€€€€€€€€€ :€€€€€€€€€ : ::::::::: : : : *=
api_implements+)lstm_8823e6b7-52ef-4b74-8f1b-b4a36a0331eb*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1308897*
go_backwards( *

time_major( :C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:EA

_output_shapes	
:А,
"
_user_specified_name
concat_1:YU
+
_output_shapes
:€€€€€€€€€ 
&
_user_specified_nameExpandDims_1:WS
+
_output_shapes
:€€€€€€€€€ 
$
_user_specified_name
ExpandDims:_
[
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

#
_user_specified_name	transpose:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_9/perm:^Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ ::6
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ :- )
'
_output_shapes
:€€€€€€€€€ 
£
ї
A__inference_lstm_layer_call_and_return_conditional_losses_1309408

inputs/
read_readvariableop_resource:	
А1
read_1_readvariableop_resource:	 А-
read_2_readvariableop_resource:	А

identity_3ИҐRead/ReadVariableOpҐRead_1/ReadVariableOpҐRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:АЉ
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference_standard_lstm_1309135v

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ h
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
а@
Ќ
*__inference_gpu_lstm_with_fallback_1310653

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:А,”
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_409036d6-1336-4fef-94b8-b56a04a88240*
api_preferred_deviceGPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
©
Щ
I__inference_sequential_1_layer_call_and_return_conditional_losses_1310403

lstm_input
lstm_1310287:	
А
lstm_1310289:	 А
lstm_1310291:	А!
dense_8_1310325: 
dense_8_1310327:!
dense_9_1310361:
dense_9_1310363:"
dense_10_1310397:
dense_10_1310399:
identityИҐ dense_10/StatefulPartitionedCallҐdense_8/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallҐlstm/StatefulPartitionedCallД
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_1310287lstm_1310289lstm_1310291*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1310286Ы
dense_8/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_8_1310325dense_8_1310327*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1310324Ю
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_1310361dense_9_1310363*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1310360Ґ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_1310397dense_10_1310399*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_1310396Е
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€®
NoOpNoOp!^dense_10/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€€€€€€€€€€
: : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:'	#
!
_user_specified_name	1310399:'#
!
_user_specified_name	1310397:'#
!
_user_specified_name	1310363:'#
!
_user_specified_name	1310361:'#
!
_user_specified_name	1310327:'#
!
_user_specified_name	1310325:'#
!
_user_specified_name	1310291:'#
!
_user_specified_name	1310289:'#
!
_user_specified_name	1310287:` \
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

$
_user_specified_name
lstm_input
єL
•
(__forward_gpu_lstm_with_fallback_1312299

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0„
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "'
concat_1_axisconcat_1/axis:output:0"
concat_1concat_1_0:output:0"#
concat_axisconcat/axis:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"%
expanddims_1ExpandDims_1:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_6fe49563-737f-4938-a2a9-c8ee445847e3*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1312124_1312300*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
Ђ
љ
A__inference_lstm_layer_call_and_return_conditional_losses_1311875
inputs_0/
read_readvariableop_resource:	
А1
read_1_readvariableop_resource:	 А-
read_2_readvariableop_resource:	А

identity_3ИҐRead/ReadVariableOpҐRead_1/ReadVariableOpҐRead_2/ReadVariableOpK
ShapeShapeinputs_0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:АЊ
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference_standard_lstm_1311602v

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ h
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

"
_user_specified_name
inputs_0
а@
Ќ
*__inference_gpu_lstm_with_fallback_1311269

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:А,”
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_8f3773d3-0e37-4317-a978-fd6c7de564f2*
api_preferred_deviceGPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
Ђ
љ
A__inference_lstm_layer_call_and_return_conditional_losses_1311448
inputs_0/
read_readvariableop_resource:	
А1
read_1_readvariableop_resource:	 А-
read_2_readvariableop_resource:	А

identity_3ИҐRead/ReadVariableOpҐRead_1/ReadVariableOpҐRead_2/ReadVariableOpK
ShapeShapeinputs_0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:АЊ
PartitionedCallPartitionedCallinputs_0zeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference_standard_lstm_1311175v

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ h
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

"
_user_specified_name
inputs_0
Ґ
ы
D__inference_dense_9_layer_call_and_return_conditional_losses_1312809

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::нѕY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
£
ї
A__inference_lstm_layer_call_and_return_conditional_losses_1312302

inputs/
read_readvariableop_resource:	
А1
read_1_readvariableop_resource:	 А-
read_2_readvariableop_resource:	А

identity_3ИҐRead/ReadVariableOpҐRead_1/ReadVariableOpҐRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:АЉ
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference_standard_lstm_1312029v

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ h
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
™Ћ
х
=__inference___backward_gpu_lstm_with_fallback_1312124_1312300
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:€€€€€€€€€ m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:€€€€€€€€€ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:€€€€€€€€€ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: Л
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕЖ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:™
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
shrink_axis_maskЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:ј
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ§
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Г
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ®
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ В
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Й
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:ў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
Г
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::нѕ≈
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::нѕ…
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Аi
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ш
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::м
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ап
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: п
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   °
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:Ј
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:Ј
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:Ј
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:Ј
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:Ј
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:Ј
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:Ј
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  з
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Аѓ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
Аµ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::“
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А÷
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesс
о:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : :€€€€€€€€€€€€€€€€€€ ::€€€€€€€€€ :€€€€€€€€€ ::€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,::€€€€€€€€€ :€€€€€€€€€ : ::::::::: : : : *=
api_implements+)lstm_6fe49563-737f-4938-a2a9-c8ee445847e3*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1312299*
go_backwards( *

time_major( :C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:EA

_output_shapes	
:А,
"
_user_specified_name
concat_1:YU
+
_output_shapes
:€€€€€€€€€ 
&
_user_specified_nameExpandDims_1:WS
+
_output_shapes
:€€€€€€€€€ 
$
_user_specified_name
ExpandDims:_
[
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

#
_user_specified_name	transpose:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_9/perm:^Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ ::6
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ :- )
'
_output_shapes
:€€€€€€€€€ 
Ґ
ы
D__inference_dense_9_layer_call_and_return_conditional_losses_1310360

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::нѕY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ц
≥
&__inference_lstm_layer_call_fn_1311010

inputs
unknown:	
А
	unknown_0:	 А
	unknown_1:	А
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1310286|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1311006:'#
!
_user_specified_name	1311004:'#
!
_user_specified_name	1311002:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
Ґ
ь
E__inference_dense_10_layer_call_and_return_conditional_losses_1312849

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::нѕY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€c
SigmoidSigmoidBiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€g
IdentityIdentitySigmoid:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
а@
Ќ
*__inference_gpu_lstm_with_fallback_1311696

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:А,”
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_9472bb29-c98e-4b95-9072-c474d6dc911f*
api_preferred_deviceGPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
 :
Ѕ
!__inference_standard_lstm_1310013

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:€€€€€€€€€А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:€€€€€€€€€Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:€€€€€€€€€АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ґ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А* 
_read_only_resource_inputs
 *
bodyR
while_body_1309928*
condR
while_cond_1309927*b
output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_321f4442-ea58-4982-a84b-60e3e3a1a34d*
api_preferred_deviceCPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
ь
µ
&__inference_lstm_layer_call_fn_1310999
inputs_0
unknown:	
А
	unknown_0:	 А
	unknown_1:	А
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1309835|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1310995:'#
!
_user_specified_name	1310993:'#
!
_user_specified_name	1310991:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

"
_user_specified_name
inputs_0
ъ*
–
while_body_1309050
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€
*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:€€€€€€€€€АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :»
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Є
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ "*
while_biasadd_biaswhile_biasadd_bias_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А:A
=

_output_shapes	
:А

_user_specified_namebias:Q	M

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
£
ї
A__inference_lstm_layer_call_and_return_conditional_losses_1312729

inputs/
read_readvariableop_resource:	
А1
read_1_readvariableop_resource:	 А-
read_2_readvariableop_resource:	А

identity_3ИҐRead/ReadVariableOpҐRead_1/ReadVariableOpҐRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:АЉ
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference_standard_lstm_1312456v

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ h
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
єL
•
(__forward_gpu_lstm_with_fallback_1312726

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0„
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "'
concat_1_axisconcat_1/axis:output:0"
concat_1concat_1_0:output:0"#
concat_axisconcat/axis:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"%
expanddims_1ExpandDims_1:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_f226f8e4-ff0b-4e39-87cf-dd08dec2a662*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1312551_1312727*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
а@
Ќ
*__inference_gpu_lstm_with_fallback_1310107

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:А,”
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_321f4442-ea58-4982-a84b-60e3e3a1a34d*
api_preferred_deviceGPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
¶
Ч
*__inference_dense_10_layer_call_fn_1312818

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_1310396|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1312814:'#
!
_user_specified_name	1312812:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Д
б
.__inference_sequential_1_layer_call_fn_1310902

lstm_input
unknown:	
А
	unknown_0:	 А
	unknown_1:	А
	unknown_2: 
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identityИҐStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_1310856|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€€€€€€€€€€
: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'	#
!
_user_specified_name	1310898:'#
!
_user_specified_name	1310896:'#
!
_user_specified_name	1310894:'#
!
_user_specified_name	1310892:'#
!
_user_specified_name	1310890:'#
!
_user_specified_name	1310888:'#
!
_user_specified_name	1310886:'#
!
_user_specified_name	1310884:'#
!
_user_specified_name	1310882:` \
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

$
_user_specified_name
lstm_input
а@
Ќ
*__inference_gpu_lstm_with_fallback_1309656

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:А,”
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_e35cb2d6-fa30-4ae4-ab3c-6ad53fcb0768*
api_preferred_deviceGPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
 :
Ѕ
!__inference_standard_lstm_1311175

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:€€€€€€€€€А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:€€€€€€€€€Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:€€€€€€€€€АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ґ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А* 
_read_only_resource_inputs
 *
bodyR
while_body_1311090*
condR
while_cond_1311089*b
output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_8f3773d3-0e37-4317-a978-fd6c7de564f2*
api_preferred_deviceCPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
ъ*
–
while_body_1311517
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€
*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:€€€€€€€€€АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :»
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Є
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ "*
while_biasadd_biaswhile_biasadd_bias_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А:A
=

_output_shapes	
:А

_user_specified_namebias:Q	M

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
™Ћ
х
=__inference___backward_gpu_lstm_with_fallback_1310654_1310830
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:€€€€€€€€€ m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:€€€€€€€€€ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:€€€€€€€€€ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: Л
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕЖ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:™
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
shrink_axis_maskЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:ј
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ§
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Г
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ®
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ В
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Й
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:ў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
Г
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::нѕ≈
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::нѕ…
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Аi
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ш
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::м
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ап
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: п
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   °
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:Ј
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:Ј
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:Ј
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:Ј
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:Ј
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:Ј
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:Ј
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  з
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Аѓ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
Аµ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::“
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А÷
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesс
о:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : :€€€€€€€€€€€€€€€€€€ ::€€€€€€€€€ :€€€€€€€€€ ::€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,::€€€€€€€€€ :€€€€€€€€€ : ::::::::: : : : *=
api_implements+)lstm_409036d6-1336-4fef-94b8-b56a04a88240*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1310829*
go_backwards( *

time_major( :C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:EA

_output_shapes	
:А,
"
_user_specified_name
concat_1:YU
+
_output_shapes
:€€€€€€€€€ 
&
_user_specified_nameExpandDims_1:WS
+
_output_shapes
:€€€€€€€€€ 
$
_user_specified_name
ExpandDims:_
[
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

#
_user_specified_name	transpose:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_9/perm:^Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ ::6
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ :- )
'
_output_shapes
:€€€€€€€€€ 
™Ћ
х
=__inference___backward_gpu_lstm_with_fallback_1311270_1311446
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:€€€€€€€€€ m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:€€€€€€€€€ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:€€€€€€€€€ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: Л
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕЖ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:™
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
shrink_axis_maskЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:ј
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ§
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Г
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ®
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ В
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Й
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:ў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
Г
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::нѕ≈
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::нѕ…
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Аi
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ш
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::м
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ап
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: п
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   °
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:Ј
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:Ј
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:Ј
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:Ј
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:Ј
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:Ј
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:Ј
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  з
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Аѓ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
Аµ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::“
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А÷
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesс
о:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : :€€€€€€€€€€€€€€€€€€ ::€€€€€€€€€ :€€€€€€€€€ ::€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,::€€€€€€€€€ :€€€€€€€€€ : ::::::::: : : : *=
api_implements+)lstm_8f3773d3-0e37-4317-a978-fd6c7de564f2*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1311445*
go_backwards( *

time_major( :C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:EA

_output_shapes	
:А,
"
_user_specified_name
concat_1:YU
+
_output_shapes
:€€€€€€€€€ 
&
_user_specified_nameExpandDims_1:WS
+
_output_shapes
:€€€€€€€€€ 
$
_user_specified_name
ExpandDims:_
[
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

#
_user_specified_name	transpose:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_9/perm:^Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ ::6
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ :- )
'
_output_shapes
:€€€€€€€€€ 
ь
µ
&__inference_lstm_layer_call_fn_1310988
inputs_0
unknown:	
А
	unknown_0:	 А
	unknown_1:	А
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1309408|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1310984:'#
!
_user_specified_name	1310982:'#
!
_user_specified_name	1310980:^ Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

"
_user_specified_name
inputs_0
©
Щ
I__inference_sequential_1_layer_call_and_return_conditional_losses_1310856

lstm_input
lstm_1310833:	
А
lstm_1310835:	 А
lstm_1310837:	А!
dense_8_1310840: 
dense_8_1310842:!
dense_9_1310845:
dense_9_1310847:"
dense_10_1310850:
dense_10_1310852:
identityИҐ dense_10/StatefulPartitionedCallҐdense_8/StatefulPartitionedCallҐdense_9/StatefulPartitionedCallҐlstm/StatefulPartitionedCallД
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_inputlstm_1310833lstm_1310835lstm_1310837*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1310832Ы
dense_8/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0dense_8_1310840dense_8_1310842*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1310324Ю
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_1310845dense_9_1310847*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1310360Ґ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall(dense_9/StatefulPartitionedCall:output:0dense_10_1310850dense_10_1310852*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_10_layer_call_and_return_conditional_losses_1310396Е
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€®
NoOpNoOp!^dense_10/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€€€€€€€€€€
: : : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:'	#
!
_user_specified_name	1310852:'#
!
_user_specified_name	1310850:'#
!
_user_specified_name	1310847:'#
!
_user_specified_name	1310845:'#
!
_user_specified_name	1310842:'#
!
_user_specified_name	1310840:'#
!
_user_specified_name	1310837:'#
!
_user_specified_name	1310835:'#
!
_user_specified_name	1310833:` \
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

$
_user_specified_name
lstm_input
єL
•
(__forward_gpu_lstm_with_fallback_1311872

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0„
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "'
concat_1_axisconcat_1/axis:output:0"
concat_1concat_1_0:output:0"#
concat_axisconcat/axis:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"%
expanddims_1ExpandDims_1:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_9472bb29-c98e-4b95-9072-c474d6dc911f*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1311697_1311873*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
ц
≥
&__inference_lstm_layer_call_fn_1311021

inputs
unknown:	
А
	unknown_0:	 А
	unknown_1:	А
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_lstm_layer_call_and_return_conditional_losses_1310832|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1311017:'#
!
_user_specified_name	1311015:'#
!
_user_specified_name	1311013:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
ъ*
–
while_body_1309928
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€
*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:€€€€€€€€€АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :»
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Є
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ "*
while_biasadd_biaswhile_biasadd_bias_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А:A
=

_output_shapes	
:А

_user_specified_namebias:Q	M

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
™Ћ
х
=__inference___backward_gpu_lstm_with_fallback_1312551_1312727
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:€€€€€€€€€ m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:€€€€€€€€€ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:€€€€€€€€€ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: Л
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕЖ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:™
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
shrink_axis_maskЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:ј
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ§
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Г
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ®
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ В
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Й
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:ў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
Г
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::нѕ≈
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::нѕ…
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Аi
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ш
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::м
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ап
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: п
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   °
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:Ј
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:Ј
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:Ј
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:Ј
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:Ј
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:Ј
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:Ј
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  з
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Аѓ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
Аµ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::“
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А÷
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesс
о:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : :€€€€€€€€€€€€€€€€€€ ::€€€€€€€€€ :€€€€€€€€€ ::€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,::€€€€€€€€€ :€€€€€€€€€ : ::::::::: : : : *=
api_implements+)lstm_f226f8e4-ff0b-4e39-87cf-dd08dec2a662*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1312726*
go_backwards( *

time_major( :C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:EA

_output_shapes	
:А,
"
_user_specified_name
concat_1:YU
+
_output_shapes
:€€€€€€€€€ 
&
_user_specified_nameExpandDims_1:WS
+
_output_shapes
:€€€€€€€€€ 
$
_user_specified_name
ExpandDims:_
[
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

#
_user_specified_name	transpose:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_9/perm:^Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ ::6
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ :- )
'
_output_shapes
:€€€€€€€€€ 
ъ*
–
while_body_1308542
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€
*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:€€€€€€€€€АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :»
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Є
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ "*
while_biasadd_biaswhile_biasadd_bias_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А:A
=

_output_shapes	
:А

_user_specified_namebias:Q	M

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
 :
Ѕ
!__inference_standard_lstm_1309135

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:€€€€€€€€€А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:€€€€€€€€€Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:€€€€€€€€€АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ґ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А* 
_read_only_resource_inputs
 *
bodyR
while_body_1309050*
condR
while_cond_1309049*b
output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_d3eb9e0d-7c67-4700-a6f4-7ba2db9f8384*
api_preferred_deviceCPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
 :
Ѕ
!__inference_standard_lstm_1312029

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:€€€€€€€€€А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:€€€€€€€€€Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:€€€€€€€€€АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ґ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А* 
_read_only_resource_inputs
 *
bodyR
while_body_1311944*
condR
while_cond_1311943*b
output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_6fe49563-737f-4938-a2a9-c8ee445847e3*
api_preferred_deviceCPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
™Ћ
х
=__inference___backward_gpu_lstm_with_fallback_1311697_1311873
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:€€€€€€€€€ m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:€€€€€€€€€ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:€€€€€€€€€ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: Л
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕЖ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:™
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
shrink_axis_maskЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:ј
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ§
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Г
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ®
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ В
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Й
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:ў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
Г
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::нѕ≈
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::нѕ…
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Аi
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ш
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::м
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ап
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: п
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   °
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:Ј
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:Ј
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:Ј
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:Ј
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:Ј
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:Ј
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:Ј
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  з
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Аѓ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
Аµ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::“
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А÷
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesс
о:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : :€€€€€€€€€€€€€€€€€€ ::€€€€€€€€€ :€€€€€€€€€ ::€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,::€€€€€€€€€ :€€€€€€€€€ : ::::::::: : : : *=
api_implements+)lstm_9472bb29-c98e-4b95-9072-c474d6dc911f*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1311872*
go_backwards( *

time_major( :C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:EA

_output_shapes	
:А,
"
_user_specified_name
concat_1:YU
+
_output_shapes
:€€€€€€€€€ 
&
_user_specified_nameExpandDims_1:WS
+
_output_shapes
:€€€€€€€€€ 
$
_user_specified_name
ExpandDims:_
[
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

#
_user_specified_name	transpose:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_9/perm:^Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ ::6
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ :- )
'
_output_shapes
:€€€€€€€€€ 
а@
Ќ
*__inference_gpu_lstm_with_fallback_1308721

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:А,”
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_8823e6b7-52ef-4b74-8f1b-b4a36a0331eb*
api_preferred_deviceGPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
љХ
“
"__inference__wrapped_model_1308981

lstm_inputA
.sequential_1_lstm_read_readvariableop_resource:	
АC
0sequential_1_lstm_read_1_readvariableop_resource:	 А?
0sequential_1_lstm_read_2_readvariableop_resource:	АH
6sequential_1_dense_8_tensordot_readvariableop_resource: B
4sequential_1_dense_8_biasadd_readvariableop_resource:H
6sequential_1_dense_9_tensordot_readvariableop_resource:B
4sequential_1_dense_9_biasadd_readvariableop_resource:I
7sequential_1_dense_10_tensordot_readvariableop_resource:C
5sequential_1_dense_10_biasadd_readvariableop_resource:
identityИҐ,sequential_1/dense_10/BiasAdd/ReadVariableOpҐ.sequential_1/dense_10/Tensordot/ReadVariableOpҐ+sequential_1/dense_8/BiasAdd/ReadVariableOpҐ-sequential_1/dense_8/Tensordot/ReadVariableOpҐ+sequential_1/dense_9/BiasAdd/ReadVariableOpҐ-sequential_1/dense_9/Tensordot/ReadVariableOpҐ%sequential_1/lstm/Read/ReadVariableOpҐ'sequential_1/lstm/Read_1/ReadVariableOpҐ'sequential_1/lstm/Read_2/ReadVariableOp_
sequential_1/lstm/ShapeShape
lstm_input*
T0*
_output_shapes
::нѕo
%sequential_1/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: q
'sequential_1/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'sequential_1/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ђ
sequential_1/lstm/strided_sliceStridedSlice sequential_1/lstm/Shape:output:0.sequential_1/lstm/strided_slice/stack:output:00sequential_1/lstm/strided_slice/stack_1:output:00sequential_1/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 sequential_1/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ©
sequential_1/lstm/zeros/packedPack(sequential_1/lstm/strided_slice:output:0)sequential_1/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:b
sequential_1/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Ґ
sequential_1/lstm/zerosFill'sequential_1/lstm/zeros/packed:output:0&sequential_1/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ d
"sequential_1/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : ≠
 sequential_1/lstm/zeros_1/packedPack(sequential_1/lstm/strided_slice:output:0+sequential_1/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_1/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_1/lstm/zeros_1Fill)sequential_1/lstm/zeros_1/packed:output:0(sequential_1/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Х
%sequential_1/lstm/Read/ReadVariableOpReadVariableOp.sequential_1_lstm_read_readvariableop_resource*
_output_shapes
:	
А*
dtype0
sequential_1/lstm/IdentityIdentity-sequential_1/lstm/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:	
АЩ
'sequential_1/lstm/Read_1/ReadVariableOpReadVariableOp0sequential_1_lstm_read_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0Г
sequential_1/lstm/Identity_1Identity/sequential_1/lstm/Read_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 АХ
'sequential_1/lstm/Read_2/ReadVariableOpReadVariableOp0sequential_1_lstm_read_2_readvariableop_resource*
_output_shapes	
:А*
dtype0
sequential_1/lstm/Identity_2Identity/sequential_1/lstm/Read_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ађ
!sequential_1/lstm/PartitionedCallPartitionedCall
lstm_input sequential_1/lstm/zeros:output:0"sequential_1/lstm/zeros_1:output:0#sequential_1/lstm/Identity:output:0%sequential_1/lstm/Identity_1:output:0%sequential_1/lstm/Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference_standard_lstm_1308627§
-sequential_1/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_8_tensordot_readvariableop_resource*
_output_shapes

: *
dtype0m
#sequential_1/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       М
$sequential_1/dense_8/Tensordot/ShapeShape*sequential_1/lstm/PartitionedCall:output:1*
T0*
_output_shapes
::нѕn
,sequential_1/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
'sequential_1/dense_8/Tensordot/GatherV2GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/free:output:05sequential_1/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
)sequential_1/dense_8/Tensordot/GatherV2_1GatherV2-sequential_1/dense_8/Tensordot/Shape:output:0,sequential_1/dense_8/Tensordot/axes:output:07sequential_1/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ≠
#sequential_1/dense_8/Tensordot/ProdProd0sequential_1/dense_8/Tensordot/GatherV2:output:0-sequential_1/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ≥
%sequential_1/dense_8/Tensordot/Prod_1Prod2sequential_1/dense_8/Tensordot/GatherV2_1:output:0/sequential_1/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
%sequential_1/dense_8/Tensordot/concatConcatV2,sequential_1/dense_8/Tensordot/free:output:0,sequential_1/dense_8/Tensordot/axes:output:03sequential_1/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Є
$sequential_1/dense_8/Tensordot/stackPack,sequential_1/dense_8/Tensordot/Prod:output:0.sequential_1/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:–
(sequential_1/dense_8/Tensordot/transpose	Transpose*sequential_1/lstm/PartitionedCall:output:1.sequential_1/dense_8/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ …
&sequential_1/dense_8/Tensordot/ReshapeReshape,sequential_1/dense_8/Tensordot/transpose:y:0-sequential_1/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€…
%sequential_1/dense_8/Tensordot/MatMulMatMul/sequential_1/dense_8/Tensordot/Reshape:output:05sequential_1/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€p
&sequential_1/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
'sequential_1/dense_8/Tensordot/concat_1ConcatV20sequential_1/dense_8/Tensordot/GatherV2:output:0/sequential_1/dense_8/Tensordot/Const_2:output:05sequential_1/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ћ
sequential_1/dense_8/TensordotReshape/sequential_1/dense_8/Tensordot/MatMul:product:00sequential_1/dense_8/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ь
+sequential_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ƒ
sequential_1/dense_8/BiasAddBiasAdd'sequential_1/dense_8/Tensordot:output:03sequential_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€З
sequential_1/dense_8/ReluRelu%sequential_1/dense_8/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€§
-sequential_1/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_1_dense_9_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0m
#sequential_1/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#sequential_1/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Й
$sequential_1/dense_9/Tensordot/ShapeShape'sequential_1/dense_8/Relu:activations:0*
T0*
_output_shapes
::нѕn
,sequential_1/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
'sequential_1/dense_9/Tensordot/GatherV2GatherV2-sequential_1/dense_9/Tensordot/Shape:output:0,sequential_1/dense_9/Tensordot/free:output:05sequential_1/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.sequential_1/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
)sequential_1/dense_9/Tensordot/GatherV2_1GatherV2-sequential_1/dense_9/Tensordot/Shape:output:0,sequential_1/dense_9/Tensordot/axes:output:07sequential_1/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$sequential_1/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ≠
#sequential_1/dense_9/Tensordot/ProdProd0sequential_1/dense_9/Tensordot/GatherV2:output:0-sequential_1/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&sequential_1/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ≥
%sequential_1/dense_9/Tensordot/Prod_1Prod2sequential_1/dense_9/Tensordot/GatherV2_1:output:0/sequential_1/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*sequential_1/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : р
%sequential_1/dense_9/Tensordot/concatConcatV2,sequential_1/dense_9/Tensordot/free:output:0,sequential_1/dense_9/Tensordot/axes:output:03sequential_1/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Є
$sequential_1/dense_9/Tensordot/stackPack,sequential_1/dense_9/Tensordot/Prod:output:0.sequential_1/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ќ
(sequential_1/dense_9/Tensordot/transpose	Transpose'sequential_1/dense_8/Relu:activations:0.sequential_1/dense_9/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€…
&sequential_1/dense_9/Tensordot/ReshapeReshape,sequential_1/dense_9/Tensordot/transpose:y:0-sequential_1/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€…
%sequential_1/dense_9/Tensordot/MatMulMatMul/sequential_1/dense_9/Tensordot/Reshape:output:05sequential_1/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€p
&sequential_1/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:n
,sequential_1/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
'sequential_1/dense_9/Tensordot/concat_1ConcatV20sequential_1/dense_9/Tensordot/GatherV2:output:0/sequential_1/dense_9/Tensordot/Const_2:output:05sequential_1/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ћ
sequential_1/dense_9/TensordotReshape/sequential_1/dense_9/Tensordot/MatMul:product:00sequential_1/dense_9/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ь
+sequential_1/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ƒ
sequential_1/dense_9/BiasAddBiasAdd'sequential_1/dense_9/Tensordot:output:03sequential_1/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€З
sequential_1/dense_9/ReluRelu%sequential_1/dense_9/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€¶
.sequential_1/dense_10/Tensordot/ReadVariableOpReadVariableOp7sequential_1_dense_10_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0n
$sequential_1/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:u
$sequential_1/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       К
%sequential_1/dense_10/Tensordot/ShapeShape'sequential_1/dense_9/Relu:activations:0*
T0*
_output_shapes
::нѕo
-sequential_1/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : У
(sequential_1/dense_10/Tensordot/GatherV2GatherV2.sequential_1/dense_10/Tensordot/Shape:output:0-sequential_1/dense_10/Tensordot/free:output:06sequential_1/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:q
/sequential_1/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ч
*sequential_1/dense_10/Tensordot/GatherV2_1GatherV2.sequential_1/dense_10/Tensordot/Shape:output:0-sequential_1/dense_10/Tensordot/axes:output:08sequential_1/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
%sequential_1/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ∞
$sequential_1/dense_10/Tensordot/ProdProd1sequential_1/dense_10/Tensordot/GatherV2:output:0.sequential_1/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: q
'sequential_1/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ґ
&sequential_1/dense_10/Tensordot/Prod_1Prod3sequential_1/dense_10/Tensordot/GatherV2_1:output:00sequential_1/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: m
+sequential_1/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ф
&sequential_1/dense_10/Tensordot/concatConcatV2-sequential_1/dense_10/Tensordot/free:output:0-sequential_1/dense_10/Tensordot/axes:output:04sequential_1/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ї
%sequential_1/dense_10/Tensordot/stackPack-sequential_1/dense_10/Tensordot/Prod:output:0/sequential_1/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ѕ
)sequential_1/dense_10/Tensordot/transpose	Transpose'sequential_1/dense_9/Relu:activations:0/sequential_1/dense_10/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ћ
'sequential_1/dense_10/Tensordot/ReshapeReshape-sequential_1/dense_10/Tensordot/transpose:y:0.sequential_1/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€ћ
&sequential_1/dense_10/Tensordot/MatMulMatMul0sequential_1/dense_10/Tensordot/Reshape:output:06sequential_1/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€q
'sequential_1/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:o
-sequential_1/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : €
(sequential_1/dense_10/Tensordot/concat_1ConcatV21sequential_1/dense_10/Tensordot/GatherV2:output:00sequential_1/dense_10/Tensordot/Const_2:output:06sequential_1/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ќ
sequential_1/dense_10/TensordotReshape0sequential_1/dense_10/Tensordot/MatMul:product:01sequential_1/dense_10/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ю
,sequential_1/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0«
sequential_1/dense_10/BiasAddBiasAdd(sequential_1/dense_10/Tensordot:output:04sequential_1/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€П
sequential_1/dense_10/SigmoidSigmoid&sequential_1/dense_10/BiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€}
IdentityIdentity!sequential_1/dense_10/Sigmoid:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€Ї
NoOpNoOp-^sequential_1/dense_10/BiasAdd/ReadVariableOp/^sequential_1/dense_10/Tensordot/ReadVariableOp,^sequential_1/dense_8/BiasAdd/ReadVariableOp.^sequential_1/dense_8/Tensordot/ReadVariableOp,^sequential_1/dense_9/BiasAdd/ReadVariableOp.^sequential_1/dense_9/Tensordot/ReadVariableOp&^sequential_1/lstm/Read/ReadVariableOp(^sequential_1/lstm/Read_1/ReadVariableOp(^sequential_1/lstm/Read_2/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€€€€€€€€€€
: : : : : : : : : 2\
,sequential_1/dense_10/BiasAdd/ReadVariableOp,sequential_1/dense_10/BiasAdd/ReadVariableOp2`
.sequential_1/dense_10/Tensordot/ReadVariableOp.sequential_1/dense_10/Tensordot/ReadVariableOp2Z
+sequential_1/dense_8/BiasAdd/ReadVariableOp+sequential_1/dense_8/BiasAdd/ReadVariableOp2^
-sequential_1/dense_8/Tensordot/ReadVariableOp-sequential_1/dense_8/Tensordot/ReadVariableOp2Z
+sequential_1/dense_9/BiasAdd/ReadVariableOp+sequential_1/dense_9/BiasAdd/ReadVariableOp2^
-sequential_1/dense_9/Tensordot/ReadVariableOp-sequential_1/dense_9/Tensordot/ReadVariableOp2N
%sequential_1/lstm/Read/ReadVariableOp%sequential_1/lstm/Read/ReadVariableOp2R
'sequential_1/lstm/Read_1/ReadVariableOp'sequential_1/lstm/Read_1/ReadVariableOp2R
'sequential_1/lstm/Read_2/ReadVariableOp'sequential_1/lstm/Read_2/ReadVariableOp:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:` \
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

$
_user_specified_name
lstm_input
§
Ц
)__inference_dense_9_layer_call_fn_1312778

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_9_layer_call_and_return_conditional_losses_1310360|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1312774:'#
!
_user_specified_name	1312772:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
а@
Ќ
*__inference_gpu_lstm_with_fallback_1312550

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:А,”
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_f226f8e4-ff0b-4e39-87cf-dd08dec2a662*
api_preferred_deviceGPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
єL
•
(__forward_gpu_lstm_with_fallback_1311445

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0„
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "'
concat_1_axisconcat_1/axis:output:0"
concat_1concat_1_0:output:0"#
concat_axisconcat/axis:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"%
expanddims_1ExpandDims_1:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_8f3773d3-0e37-4317-a978-fd6c7de564f2*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1311270_1311446*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
єL
•
(__forward_gpu_lstm_with_fallback_1309405

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0„
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "'
concat_1_axisconcat_1/axis:output:0"
concat_1concat_1_0:output:0"#
concat_axisconcat/axis:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"%
expanddims_1ExpandDims_1:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_d3eb9e0d-7c67-4700-a6f4-7ba2db9f8384*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1309230_1309406*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
а@
Ќ
*__inference_gpu_lstm_with_fallback_1312123

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4Иc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : o

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : s
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes	
:А,”
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_6fe49563-737f-4938-a2a9-c8ee445847e3*
api_preferred_deviceGPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
°

∆
while_cond_1311516
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1311516___redundant_placeholder05
1while_while_cond_1311516___redundant_placeholder15
1while_while_cond_1311516___redundant_placeholder25
1while_while_cond_1311516___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
ъ*
–
while_body_1311090
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€
*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:€€€€€€€€€АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :»
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Є
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ "*
while_biasadd_biaswhile_biasadd_bias_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А:A
=

_output_shapes	
:А

_user_specified_namebias:Q	M

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
Ґ
ы
D__inference_dense_8_layer_call_and_return_conditional_losses_1312769

inputs3
!tensordot_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::нѕY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€]
ReluReluBiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€n
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
ъ*
–
while_body_1312371
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€
*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:€€€€€€€€€АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :»
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Є
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ "*
while_biasadd_biaswhile_biasadd_bias_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А:A
=

_output_shapes	
:А

_user_specified_namebias:Q	M

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
£
ї
A__inference_lstm_layer_call_and_return_conditional_losses_1309835

inputs/
read_readvariableop_resource:	
А1
read_1_readvariableop_resource:	 А-
read_2_readvariableop_resource:	А

identity_3ИҐRead/ReadVariableOpҐRead_1/ReadVariableOpҐRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:АЉ
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference_standard_lstm_1309562v

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ h
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
 :
Ѕ
!__inference_standard_lstm_1311602

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:€€€€€€€€€А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:€€€€€€€€€Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:€€€€€€€€€АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ґ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А* 
_read_only_resource_inputs
 *
bodyR
while_body_1311517*
condR
while_cond_1311516*b
output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_9472bb29-c98e-4b95-9072-c474d6dc911f*
api_preferred_deviceCPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
Ґ
ь
E__inference_dense_10_layer_call_and_return_conditional_losses_1310396

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::нѕY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : њ
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : І
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€c
SigmoidSigmoidBiasAdd:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€g
IdentityIdentitySigmoid:y:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
‘
Ў
%__inference_signature_wrapper_1310977

lstm_input
unknown:	
А
	unknown_0:	 А
	unknown_1:	А
	unknown_2: 
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_1308981|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€€€€€€€€€€
: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'	#
!
_user_specified_name	1310973:'#
!
_user_specified_name	1310971:'#
!
_user_specified_name	1310969:'#
!
_user_specified_name	1310967:'#
!
_user_specified_name	1310965:'#
!
_user_specified_name	1310963:'#
!
_user_specified_name	1310961:'#
!
_user_specified_name	1310959:'#
!
_user_specified_name	1310957:` \
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

$
_user_specified_name
lstm_input
™Ћ
х
=__inference___backward_gpu_lstm_with_fallback_1310108_1310284
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnA
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm)
%gradients_squeeze_grad_shape_cudnnrnn+
'gradients_squeeze_1_grad_shape_cudnnrnn!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5И^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:€€€€€€€€€ m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ `
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:€€€€€€€€€ `
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:€€€€€€€€€ O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: Л
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕЖ
3gradients/strided_slice_grad/StridedSliceGrad/beginConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€{
1gradients/strided_slice_grad/StridedSliceGrad/endConst*
_output_shapes
:*
dtype0*
valueB: 
5gradients/strided_slice_grad/StridedSliceGrad/stridesConst*
_output_shapes
:*
dtype0*
valueB:™
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0<gradients/strided_slice_grad/StridedSliceGrad/begin:output:0:gradients/strided_slice_grad/StridedSliceGrad/end:output:0>gradients/strided_slice_grad/StridedSliceGrad/strides:output:0gradients/grad_ys_0:output:0*
Index0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
shrink_axis_maskЬ
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:ј
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
gradients/Squeeze_grad/ShapeShape%gradients_squeeze_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ§
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_2:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Г
gradients/Squeeze_1_grad/ShapeShape'gradients_squeeze_1_grad_shape_cudnnrnn*
T0*
_output_shapes
::нѕ®
 gradients/Squeeze_1_grad/ReshapeReshapegradients/grad_ys_3:output:0'gradients/Squeeze_1_grad/Shape:output:0*
T0*+
_output_shapes
:€€€€€€€€€ В
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
N*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:Й
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn%gradients_squeeze_grad_shape_cudnnrnn'gradients_squeeze_1_grad_shape_cudnnrnngradients/AddN:sum:0'gradients/Squeeze_grad/Reshape:output:0)gradients/Squeeze_1_grad/Reshape:output:0gradients_zeros_like_cudnnrnn*
T0*i
_output_shapesW
U:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,Ц
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:ў
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
Г
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
::нѕ≈
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ З
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
::нѕ…
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Ъ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:јj
gradients/concat_1_grad/Shape_4Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_5Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
dtype0*
valueB:Аj
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
dtype0*
valueB:Аi
gradients/concat_1_grad/Shape_8Const*
_output_shapes
:*
dtype0*
valueB: i
gradients/concat_1_grad/Shape_9Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_10Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_13Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
dtype0*
valueB: j
 gradients/concat_1_grad/Shape_15Const*
_output_shapes
:*
dtype0*
valueB: ш
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::м
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:јр
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:Ар
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:Ап
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
: п
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
: т
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
: m
gradients/Reshape_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   °
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_2_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_3_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"    
   І
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

: 
o
gradients/Reshape_4_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_5_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_6_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:  o
gradients/Reshape_7_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"        І
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:  h
gradients/Reshape_8_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
: h
gradients/Reshape_9_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: £
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_10_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_12_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_14_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
: i
gradients/Reshape_15_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB: ¶
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
: Ь
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:µ
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:Ј
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:Ј
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:Ј
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:
 Ь
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:Ј
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:Ј
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:Ј
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  Ь
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:Ј
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:  з
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
T0*
_output_shapes	
:Аѓ
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
T0*
_output_shapes
:	
Аµ
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
T0*
_output_shapes
:	 А\
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :Т
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:Аh
gradients/concat_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:А 
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::“
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:А÷
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:А{
IdentityIdentity&gradients/transpose_grad/transpose:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
t

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ v

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_3Identity$gradients/split_grad/concat:output:0*
T0*
_output_shapes
:	
Аh

Identity_4Identity&gradients/split_1_grad/concat:output:0*
T0*
_output_shapes
:	 Аd

Identity_5Identity&gradients/concat_grad/Slice_1:output:0*
T0*
_output_shapes	
:А"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesс
о:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : :€€€€€€€€€€€€€€€€€€ ::€€€€€€€€€ :€€€€€€€€€ ::€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :А,::€€€€€€€€€ :€€€€€€€€€ : ::::::::: : : : *=
api_implements+)lstm_321f4442-ea58-4982-a84b-60e3e3a1a34d*
api_preferred_deviceGPU*C
forward_function_name*(__forward_gpu_lstm_with_fallback_1310283*
go_backwards( *

time_major( :C?

_output_shapes
: 
%
_user_specified_nameconcat/axis:IE

_output_shapes
: 
+
_user_specified_namesplit_1/split_dim:GC

_output_shapes
: 
)
_user_specified_namesplit/split_dim:IE

_output_shapes
: 
+
_user_specified_namesplit_2/split_dim:LH

_output_shapes
:
*
_user_specified_nametranspose_8/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_7/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_6/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_5/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_4/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_3/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_2/perm:LH

_output_shapes
:
*
_user_specified_nametranspose_1/perm:EA

_output_shapes
: 
'
_user_specified_nameconcat_1/axis:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:JF

_output_shapes
:
(
_user_specified_nametranspose/perm:EA

_output_shapes	
:А,
"
_user_specified_name
concat_1:YU
+
_output_shapes
:€€€€€€€€€ 
&
_user_specified_nameExpandDims_1:WS
+
_output_shapes
:€€€€€€€€€ 
$
_user_specified_name
ExpandDims:_
[
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

#
_user_specified_name	transpose:B	>

_output_shapes
:
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:UQ
+
_output_shapes
:€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:LH

_output_shapes
:
*
_user_specified_nametranspose_9/perm:^Z
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
"
_user_specified_name
CudnnRNN:

_output_shapes
: :-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ ::6
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ :- )
'
_output_shapes
:€€€€€€€€€ 
°

∆
while_cond_1310473
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1310473___redundant_placeholder05
1while_while_cond_1310473___redundant_placeholder15
1while_while_cond_1310473___redundant_placeholder25
1while_while_cond_1310473___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
°

∆
while_cond_1309927
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1309927___redundant_placeholder05
1while_while_cond_1309927___redundant_placeholder15
1while_while_cond_1309927___redundant_placeholder25
1while_while_cond_1309927___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
°

∆
while_cond_1308541
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1308541___redundant_placeholder05
1while_while_cond_1308541___redundant_placeholder15
1while_while_cond_1308541___redundant_placeholder25
1while_while_cond_1308541___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
єL
•
(__forward_gpu_lstm_with_fallback_1310283

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0„
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "'
concat_1_axisconcat_1/axis:output:0"
concat_1concat_1_0:output:0"#
concat_axisconcat/axis:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"%
expanddims_1ExpandDims_1:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_321f4442-ea58-4982-a84b-60e3e3a1a34d*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1310108_1310284*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
єL
•
(__forward_gpu_lstm_with_fallback_1310829

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0„
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "'
concat_1_axisconcat_1/axis:output:0"
concat_1concat_1_0:output:0"#
concat_axisconcat/axis:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"%
expanddims_1ExpandDims_1:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_409036d6-1336-4fef-94b8-b56a04a88240*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1310654_1310830*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
ъ*
–
while_body_1309477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_matmul_kernel_0%
!while_matmul_1_recurrent_kernel_0
while_biasadd_bias_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_matmul_kernel#
while_matmul_1_recurrent_kernel
while_biasadd_biasИ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   ¶
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:€€€€€€€€€
*
element_dtype0Т
while/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while_matmul_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€АГ
while/MatMul_1MatMulwhile_placeholder_2!while_matmul_1_recurrent_kernel_0*
T0*(
_output_shapes
:€€€€€€€€€Аw
	while/addAddV2while/MatMul:product:0while/MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€Аp
while/BiasAddBiasAddwhile/add:z:0while_biasadd_bias_0*
T0*(
_output_shapes
:€€€€€€€€€АW
while/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :»
while/splitSplitwhile/split/split_dim:output:0while/BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_split`
while/SigmoidSigmoidwhile/split:output:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_1Sigmoidwhile/split:output:1*
T0*'
_output_shapes
:€€€€€€€€€ l
	while/mulMulwhile/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:€€€€€€€€€ Z

while/TanhTanhwhile/split:output:2*
T0*'
_output_shapes
:€€€€€€€€€ g
while/mul_1Mulwhile/Sigmoid:y:0while/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ f
while/add_1AddV2while/mul:z:0while/mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ b
while/Sigmoid_2Sigmoidwhile/split:output:3*
T0*'
_output_shapes
:€€€€€€€€€ W
while/Tanh_1Tanhwhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ k
while/mul_2Mulwhile/Sigmoid_2:y:0while/Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ Є
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/mul_2:z:0*
_output_shapes
: *
element_dtype0:йи“O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_2AddV2while_placeholderwhile/add_2/y:output:0*
T0*
_output_shapes
: O
while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_3AddV2while_while_loop_counterwhile/add_3/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_3:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: N
while/Identity_2Identitywhile/add_2:z:0*
T0*
_output_shapes
: y
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: _
while/Identity_4Identitywhile/mul_2:z:0*
T0*'
_output_shapes
:€€€€€€€€€ _
while/Identity_5Identitywhile/add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ "*
while_biasadd_biaswhile_biasadd_bias_0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0")
while_identitywhile/Identity:output:0"D
while_matmul_1_recurrent_kernel!while_matmul_1_recurrent_kernel_0",
while_matmul_kernelwhile_matmul_kernel_0",
while_strided_slicewhile_strided_slice_0"®
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А:A
=

_output_shapes	
:А

_user_specified_namebias:Q	M

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
 :
Ѕ
!__inference_standard_lstm_1310559

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:€€€€€€€€€А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:€€€€€€€€€Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:€€€€€€€€€АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ґ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А* 
_read_only_resource_inputs
 *
bodyR
while_body_1310474*
condR
while_cond_1310473*b
output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_409036d6-1336-4fef-94b8-b56a04a88240*
api_preferred_deviceCPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
єL
•
(__forward_gpu_lstm_with_fallback_1308897

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0„
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "'
concat_1_axisconcat_1/axis:output:0"
concat_1concat_1_0:output:0"#
concat_axisconcat/axis:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"%
expanddims_1ExpandDims_1:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_8823e6b7-52ef-4b74-8f1b-b4a36a0331eb*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1308722_1308898*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
ЛЪ
 
#__inference__traced_restore_1313177
file_prefix1
assignvariableop_dense_8_kernel: -
assignvariableop_1_dense_8_bias:3
!assignvariableop_2_dense_9_kernel:-
assignvariableop_3_dense_9_bias:4
"assignvariableop_4_dense_10_kernel:.
 assignvariableop_5_dense_10_bias:;
(assignvariableop_6_lstm_lstm_cell_kernel:	
АE
2assignvariableop_7_lstm_lstm_cell_recurrent_kernel:	 А5
&assignvariableop_8_lstm_lstm_cell_bias:	А&
assignvariableop_9_iteration:	 +
!assignvariableop_10_learning_rate: C
0assignvariableop_11_adam_m_lstm_lstm_cell_kernel:	
АC
0assignvariableop_12_adam_v_lstm_lstm_cell_kernel:	
АM
:assignvariableop_13_adam_m_lstm_lstm_cell_recurrent_kernel:	 АM
:assignvariableop_14_adam_v_lstm_lstm_cell_recurrent_kernel:	 А=
.assignvariableop_15_adam_m_lstm_lstm_cell_bias:	А=
.assignvariableop_16_adam_v_lstm_lstm_cell_bias:	А;
)assignvariableop_17_adam_m_dense_8_kernel: ;
)assignvariableop_18_adam_v_dense_8_kernel: 5
'assignvariableop_19_adam_m_dense_8_bias:5
'assignvariableop_20_adam_v_dense_8_bias:;
)assignvariableop_21_adam_m_dense_9_kernel:;
)assignvariableop_22_adam_v_dense_9_kernel:5
'assignvariableop_23_adam_m_dense_9_bias:5
'assignvariableop_24_adam_v_dense_9_bias:<
*assignvariableop_25_adam_m_dense_10_kernel:<
*assignvariableop_26_adam_v_dense_10_kernel:6
(assignvariableop_27_adam_m_dense_10_bias:6
(assignvariableop_28_adam_v_dense_10_bias:%
assignvariableop_29_total_1: %
assignvariableop_30_count_1: #
assignvariableop_31_total: #
assignvariableop_32_count: 
identity_34ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9≠
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*”
value…B∆"B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHі
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ћ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_9_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_9_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:є
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_10_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_10_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_6AssignVariableOp(assignvariableop_6_lstm_lstm_cell_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_7AssignVariableOp2assignvariableop_7_lstm_lstm_cell_recurrent_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_8AssignVariableOp&assignvariableop_8_lstm_lstm_cell_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0	*
_output_shapes
:≥
AssignVariableOp_9AssignVariableOpassignvariableop_9_iterationIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_10AssignVariableOp!assignvariableop_10_learning_rateIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_11AssignVariableOp0assignvariableop_11_adam_m_lstm_lstm_cell_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_12AssignVariableOp0assignvariableop_12_adam_v_lstm_lstm_cell_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_13AssignVariableOp:assignvariableop_13_adam_m_lstm_lstm_cell_recurrent_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_14AssignVariableOp:assignvariableop_14_adam_v_lstm_lstm_cell_recurrent_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_15AssignVariableOp.assignvariableop_15_adam_m_lstm_lstm_cell_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_16AssignVariableOp.assignvariableop_16_adam_v_lstm_lstm_cell_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_m_dense_8_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_v_dense_8_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_m_dense_8_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_v_dense_8_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_m_dense_9_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_v_dense_9_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_m_dense_9_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_v_dense_9_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_m_dense_10_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_v_dense_10_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_m_dense_10_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_v_dense_10_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 •
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: о
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_34Identity_34:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%!!

_user_specified_namecount:% !

_user_specified_nametotal:'#
!
_user_specified_name	count_1:'#
!
_user_specified_name	total_1:40
.
_user_specified_nameAdam/v/dense_10/bias:40
.
_user_specified_nameAdam/m/dense_10/bias:62
0
_user_specified_nameAdam/v/dense_10/kernel:62
0
_user_specified_nameAdam/m/dense_10/kernel:3/
-
_user_specified_nameAdam/v/dense_9/bias:3/
-
_user_specified_nameAdam/m/dense_9/bias:51
/
_user_specified_nameAdam/v/dense_9/kernel:51
/
_user_specified_nameAdam/m/dense_9/kernel:3/
-
_user_specified_nameAdam/v/dense_8/bias:3/
-
_user_specified_nameAdam/m/dense_8/bias:51
/
_user_specified_nameAdam/v/dense_8/kernel:51
/
_user_specified_nameAdam/m/dense_8/kernel::6
4
_user_specified_nameAdam/v/lstm/lstm_cell/bias::6
4
_user_specified_nameAdam/m/lstm/lstm_cell/bias:FB
@
_user_specified_name(&Adam/v/lstm/lstm_cell/recurrent_kernel:FB
@
_user_specified_name(&Adam/m/lstm/lstm_cell/recurrent_kernel:<8
6
_user_specified_nameAdam/v/lstm/lstm_cell/kernel:<8
6
_user_specified_nameAdam/m/lstm/lstm_cell/kernel:-)
'
_user_specified_namelearning_rate:)
%
#
_user_specified_name	iteration:3	/
-
_user_specified_namelstm/lstm_cell/bias:?;
9
_user_specified_name!lstm/lstm_cell/recurrent_kernel:51
/
_user_specified_namelstm/lstm_cell/kernel:-)
'
_user_specified_namedense_10/bias:/+
)
_user_specified_namedense_10/kernel:,(
&
_user_specified_namedense_9/bias:.*
(
_user_specified_namedense_9/kernel:,(
&
_user_specified_namedense_8/bias:.*
(
_user_specified_namedense_8/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
°

∆
while_cond_1311089
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice5
1while_while_cond_1311089___redundant_placeholder05
1while_while_cond_1311089___redundant_placeholder15
1while_while_cond_1311089___redundant_placeholder25
1while_while_cond_1311089___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :€€€€€€€€€ :€€€€€€€€€ : :::::


_output_shapes
::	

_output_shapes
::

_output_shapes
::

_output_shapes
::EA

_output_shapes
: 
'
_user_specified_namestrided_slice:-)
'
_output_shapes
:€€€€€€€€€ :-)
'
_output_shapes
:€€€€€€€€€ :

_output_shapes
: :

_output_shapes
: :PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter
єL
•
(__forward_gpu_lstm_with_fallback_1309832

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
transpose_9_perm

cudnnrnn_0

cudnnrnn_1

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisИc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T0P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : u
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€ Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
splitSplitsplit/split_dim:output:0kernel*
T0*<
_output_shapes*
(:
 :
 :
 :
 *
	num_splitS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ц
split_1Splitsplit_1/split_dim:output:0recurrent_kernel*
T0*<
_output_shapes*
(:  :  :  :  *
	num_splitY

zeros_likeConst*
_output_shapes	
:А*
dtype0*
valueBА*    M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : r
concatConcatV2zeros_like:output:0biasconcat/axis:output:0*
N*
T0*
_output_shapes	
:АS
split_2/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Э
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*D
_output_shapes2
0: : : : : : : : *
	num_splitX
ConstConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0*
_output_shapes

: 
Y
ReshapeReshapetranspose_1:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_1Reshapetranspose_2:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_2Reshapetranspose_3:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       l
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0*
_output_shapes

: 
[
	Reshape_3Reshapetranspose_4:y:0Const:output:0*
T0*
_output_shapes	
:јa
transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_4Reshapetranspose_5:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_6/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_5Reshapetranspose_6:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_7/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_6Reshapetranspose_7:y:0Const:output:0*
T0*
_output_shapes	
:Аa
transpose_8/permConst*
_output_shapes
:*
dtype0*
valueB"       n
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0*
_output_shapes

:  [
	Reshape_7Reshapetranspose_8:y:0Const:output:0*
T0*
_output_shapes	
:А[
	Reshape_8Reshapesplit_2:output:0Const:output:0*
T0*
_output_shapes
: [
	Reshape_9Reshapesplit_2:output:1Const:output:0*
T0*
_output_shapes
: \

Reshape_10Reshapesplit_2:output:2Const:output:0*
T0*
_output_shapes
: \

Reshape_11Reshapesplit_2:output:3Const:output:0*
T0*
_output_shapes
: \

Reshape_12Reshapesplit_2:output:4Const:output:0*
T0*
_output_shapes
: \

Reshape_13Reshapesplit_2:output:5Const:output:0*
T0*
_output_shapes
: \

Reshape_14Reshapesplit_2:output:6Const:output:0*
T0*
_output_shapes
: \

Reshape_15Reshapesplit_2:output:7Const:output:0*
T0*
_output_shapes
: O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0„
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0*f
_output_shapesT
R:€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :f
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_9/permConst*
_output_shapes
:*
dtype0*!
valueB"          Е
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ p
SqueezeSqueezeCudnnRNN:output_h:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 r
	Squeeze_1SqueezeCudnnRNN:output_c:0*
T0*'
_output_shapes
:€€€€€€€€€ *
squeeze_dims
 [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *   @^
IdentityIdentitystrided_slice:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_9:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ Z

Identity_2IdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€ \

Identity_3IdentitySqueeze_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "'
concat_1_axisconcat_1/axis:output:0"
concat_1concat_1_0:output:0"#
concat_axisconcat/axis:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"%
expanddims_1ExpandDims_1:output:0"!

expanddimsExpandDims:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"
init_cinit_c_0"
init_hinit_h_0"/
split_1_split_dimsplit_1/split_dim:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"-
transpose_2_permtranspose_2/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"-
transpose_5_permtranspose_5/perm:output:0"-
transpose_6_permtranspose_6/perm:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_8_permtranspose_8/perm:output:0"-
transpose_9_permtranspose_9/perm:output:0")
transpose_permtranspose/perm:output:0"
	transposetranspose_0:y:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_e35cb2d6-fa30-4ae4-ab3c-6ad53fcb0768*
api_preferred_deviceGPU*Y
backward_function_name?=__inference___backward_gpu_lstm_with_fallback_1309657_1309833*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
 :
Ѕ
!__inference_standard_lstm_1308627

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:€€€€€€€€€А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:€€€€€€€€€Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:€€€€€€€€€АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ґ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А* 
_read_only_resource_inputs
 *
bodyR
while_body_1308542*
condR
while_cond_1308541*b
output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_8823e6b7-52ef-4b74-8f1b-b4a36a0331eb*
api_preferred_deviceCPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
 :
Ѕ
!__inference_standard_lstm_1312456

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€
P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“Ж
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€
   а
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:й
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€
*
shrink_axis_maske
MatMulMatMulstrided_slice_1:output:0kernel*
T0*(
_output_shapes
:€€€€€€€€€А_
MatMul_1MatMulinit_hrecurrent_kernel*
T0*(
_output_shapes
:€€€€€€€€€Аe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:€€€€€€€€€АT
BiasAddBiasAddadd:z:0bias*
T0*(
_output_shapes
:€€€€€€€€€АQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ґ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ *
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:€€€€€€€€€ S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:€€€€€€€€€ N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:€€€€€€€€€ U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€ T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:€€€€€€€€€ K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€ Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€ n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    ґ
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:йи“F
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ы
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*c
_output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А* 
_read_only_resource_inputs
 *
bodyR
while_body_1312371*
condR
while_cond_1312370*b
output_shapesQ
O: : : : :€€€€€€€€€ :€€€€€€€€€ : : :	
А:	 А:А*
parallel_iterations Б
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€    Ћ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ *
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:З
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€ *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Я
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?`
IdentityIdentitystrided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€ f

Identity_1Identitytranspose_1:y:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ X

Identity_2Identitywhile:output:4*
T0*'
_output_shapes
:€€€€€€€€€ X

Identity_3Identitywhile:output:5*
T0*'
_output_shapes
:€€€€€€€€€ I

Identity_4Identityruntime:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*v
_input_shapese
c:€€€€€€€€€€€€€€€€€€
:€€€€€€€€€ :€€€€€€€€€ :	
А:	 А:А*=
api_implements+)lstm_f226f8e4-ff0b-4e39-87cf-dd08dec2a662*
api_preferred_deviceCPU*
go_backwards( *

time_major( :A=

_output_shapes	
:А

_user_specified_namebias:QM

_output_shapes
:	 А
*
_user_specified_namerecurrent_kernel:GC

_output_shapes
:	
А
 
_user_specified_namekernel:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_c:OK
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinit_h:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
Д
б
.__inference_sequential_1_layer_call_fn_1310879

lstm_input
unknown:	
А
	unknown_0:	 А
	unknown_1:	А
	unknown_2: 
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identityИҐStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_1310403|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:€€€€€€€€€€€€€€€€€€
: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'	#
!
_user_specified_name	1310875:'#
!
_user_specified_name	1310873:'#
!
_user_specified_name	1310871:'#
!
_user_specified_name	1310869:'#
!
_user_specified_name	1310867:'#
!
_user_specified_name	1310865:'#
!
_user_specified_name	1310863:'#
!
_user_specified_name	1310861:'#
!
_user_specified_name	1310859:` \
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

$
_user_specified_name
lstm_input
£
ї
A__inference_lstm_layer_call_and_return_conditional_losses_1310286

inputs/
read_readvariableop_resource:	
А1
read_1_readvariableop_resource:	 А-
read_2_readvariableop_resource:	А

identity_3ИҐRead/ReadVariableOpҐRead_1/ReadVariableOpҐRead_2/ReadVariableOpI
ShapeShapeinputs*
T0*
_output_shapes
::нѕ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€ q
Read/ReadVariableOpReadVariableOpread_readvariableop_resource*
_output_shapes
:	
А*
dtype0[
IdentityIdentityRead/ReadVariableOp:value:0*
T0*
_output_shapes
:	
Аu
Read_1/ReadVariableOpReadVariableOpread_1_readvariableop_resource*
_output_shapes
:	 А*
dtype0_

Identity_1IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	 Аq
Read_2/ReadVariableOpReadVariableOpread_2_readvariableop_resource*
_output_shapes	
:А*
dtype0[

Identity_2IdentityRead_2/ReadVariableOp:value:0*
T0*
_output_shapes	
:АЉ
PartitionedCallPartitionedCallinputszeros:output:0zeros_1:output:0Identity:output:0Identity_1:output:0Identity_2:output:0*
Tin

2*
Tout	
2*
_collective_manager_ids
 *o
_output_shapes]
[:€€€€€€€€€ :€€€€€€€€€€€€€€€€€€ :€€€€€€€€€ :€€€€€€€€€ : * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference_standard_lstm_1310013v

Identity_3IdentityPartitionedCall:output:1^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ h
NoOpNoOp^Read/ReadVariableOp^Read_1/ReadVariableOp^Read_2/ReadVariableOp*
_output_shapes
 "!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€€€€€€€€€€
: : : 2*
Read/ReadVariableOpRead/ReadVariableOp2.
Read_1/ReadVariableOpRead_1/ReadVariableOp2.
Read_2/ReadVariableOpRead_2/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€

 
_user_specified_nameinputs
§
Ц
)__inference_dense_8_layer_call_fn_1312738

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_8_layer_call_and_return_conditional_losses_1310324|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	1312734:'#
!
_user_specified_name	1312732:\ X
4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs" L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ћ
serving_defaultЈ
N

lstm_input@
serving_default_lstm_input:0€€€€€€€€€€€€€€€€€€
I
dense_10=
StatefulPartitionedCall:0€€€€€€€€€€€€€€€€€€tensorflow/serving/predict:ЗЯ
В
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
Џ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
ї
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ї
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
ї
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
_
/0
01
12
3
4
%5
&6
-7
.8"
trackable_list_wrapper
_
/0
01
12
3
4
%5
&6
-7
.8"
trackable_list_wrapper
 "
trackable_list_wrapper
 
2non_trainable_variables

3layers
4metrics
5layer_regularization_losses
6layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
ѕ
7trace_0
8trace_12Ш
.__inference_sequential_1_layer_call_fn_1310879
.__inference_sequential_1_layer_call_fn_1310902µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z7trace_0z8trace_1
Е
9trace_0
:trace_12ќ
I__inference_sequential_1_layer_call_and_return_conditional_losses_1310403
I__inference_sequential_1_layer_call_and_return_conditional_losses_1310856µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z9trace_0z:trace_1
–BЌ
"__inference__wrapped_model_1308981
lstm_input"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ь
;
_variables
<_iterations
=_learning_rate
>_index_dict
?
_momentums
@_velocities
A_update_step_xla"
experimentalOptimizer
,
Bserving_default"
signature_map
5
/0
01
12"
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
є

Cstates
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ў
Itrace_0
Jtrace_1
Ktrace_2
Ltrace_32н
&__inference_lstm_layer_call_fn_1310988
&__inference_lstm_layer_call_fn_1310999
&__inference_lstm_layer_call_fn_1311010
&__inference_lstm_layer_call_fn_1311021 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zItrace_0zJtrace_1zKtrace_2zLtrace_3
ƒ
Mtrace_0
Ntrace_1
Otrace_2
Ptrace_32ў
A__inference_lstm_layer_call_and_return_conditional_losses_1311448
A__inference_lstm_layer_call_and_return_conditional_losses_1311875
A__inference_lstm_layer_call_and_return_conditional_losses_1312302
A__inference_lstm_layer_call_and_return_conditional_losses_1312729 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zMtrace_0zNtrace_1zOtrace_2zPtrace_3
"
_generic_user_object
ш
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
U__call__
*V&call_and_return_all_conditional_losses
W_random_generator
X
state_size

/kernel
0recurrent_kernel
1bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
г
^trace_02∆
)__inference_dense_8_layer_call_fn_1312738Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z^trace_0
ю
_trace_02б
D__inference_dense_8_layer_call_and_return_conditional_losses_1312769Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z_trace_0
 : 2dense_8/kernel
:2dense_8/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
г
etrace_02∆
)__inference_dense_9_layer_call_fn_1312778Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zetrace_0
ю
ftrace_02б
D__inference_dense_9_layer_call_and_return_conditional_losses_1312809Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zftrace_0
 :2dense_9/kernel
:2dense_9/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
д
ltrace_02«
*__inference_dense_10_layer_call_fn_1312818Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zltrace_0
€
mtrace_02в
E__inference_dense_10_layer_call_and_return_conditional_losses_1312849Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zmtrace_0
!:2dense_10/kernel
:2dense_10/bias
(:&	
А2lstm/lstm_cell/kernel
2:0	 А2lstm/lstm_cell/recurrent_kernel
": А2lstm/lstm_cell/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
щBц
.__inference_sequential_1_layer_call_fn_1310879
lstm_input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
.__inference_sequential_1_layer_call_fn_1310902
lstm_input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ФBС
I__inference_sequential_1_layer_call_and_return_conditional_losses_1310403
lstm_input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ФBС
I__inference_sequential_1_layer_call_and_return_conditional_losses_1310856
lstm_input"µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
∞
<0
p1
q2
r3
s4
t5
u6
v7
w8
x9
y10
z11
{12
|13
}14
~15
16
А17
Б18"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
`
p0
r1
t2
v3
x4
z5
|6
~7
А8"
trackable_list_wrapper
`
q0
s1
u2
w3
y4
{5
}6
7
Б8"
trackable_list_wrapper
µ2≤ѓ
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
ѕBћ
%__inference_signature_wrapper_1310977
lstm_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ДBБ
&__inference_lstm_layer_call_fn_1310988inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ДBБ
&__inference_lstm_layer_call_fn_1310999inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
&__inference_lstm_layer_call_fn_1311010inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ВB€
&__inference_lstm_layer_call_fn_1311021inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЯBЬ
A__inference_lstm_layer_call_and_return_conditional_losses_1311448inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЯBЬ
A__inference_lstm_layer_call_and_return_conditional_losses_1311875inputs_0" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЭBЪ
A__inference_lstm_layer_call_and_return_conditional_losses_1312302inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЭBЪ
A__inference_lstm_layer_call_and_return_conditional_losses_1312729inputs" 
√≤њ
FullArgSpec:
args2Ъ/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsҐ

 
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
5
/0
01
12"
trackable_list_wrapper
5
/0
01
12"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
Q	variables
Rtrainable_variables
Sregularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
є2ґ≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
є2ґ≥
ђ≤®
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsҐ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_dense_8_layer_call_fn_1312738inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_dense_8_layer_call_and_return_conditional_losses_1312769inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
”B–
)__inference_dense_9_layer_call_fn_1312778inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
D__inference_dense_9_layer_call_and_return_conditional_losses_1312809inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
‘B—
*__inference_dense_10_layer_call_fn_1312818inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
E__inference_dense_10_layer_call_and_return_conditional_losses_1312849inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
R
З	variables
И	keras_api

Йtotal

Кcount"
_tf_keras_metric
c
Л	variables
М	keras_api

Нtotal

Оcount
П
_fn_kwargs"
_tf_keras_metric
-:+	
А2Adam/m/lstm/lstm_cell/kernel
-:+	
А2Adam/v/lstm/lstm_cell/kernel
7:5	 А2&Adam/m/lstm/lstm_cell/recurrent_kernel
7:5	 А2&Adam/v/lstm/lstm_cell/recurrent_kernel
':%А2Adam/m/lstm/lstm_cell/bias
':%А2Adam/v/lstm/lstm_cell/bias
%:# 2Adam/m/dense_8/kernel
%:# 2Adam/v/dense_8/kernel
:2Adam/m/dense_8/bias
:2Adam/v/dense_8/bias
%:#2Adam/m/dense_9/kernel
%:#2Adam/v/dense_9/kernel
:2Adam/m/dense_9/bias
:2Adam/v/dense_9/bias
&:$2Adam/m/dense_10/kernel
&:$2Adam/v/dense_10/kernel
 :2Adam/m/dense_10/bias
 :2Adam/v/dense_10/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
Й0
К1"
trackable_list_wrapper
.
З	variables"
_generic_user_object
:  (2total
:  (2count
0
Н0
О1"
trackable_list_wrapper
.
Л	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperґ
"__inference__wrapped_model_1308981П	/01%&-.@Ґ=
6Ґ3
1К.

lstm_input€€€€€€€€€€€€€€€€€€

™ "@™=
;
dense_10/К,
dense_10€€€€€€€€€€€€€€€€€€∆
E__inference_dense_10_layer_call_and_return_conditional_losses_1312849}-.<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€€€€€€€€€€
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ †
*__inference_dense_10_layer_call_fn_1312818r-.<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€€€€€€€€€€
™ ".К+
unknown€€€€€€€€€€€€€€€€€€≈
D__inference_dense_8_layer_call_and_return_conditional_losses_1312769}<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€€€€€€€€€€ 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ Я
)__inference_dense_8_layer_call_fn_1312738r<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€€€€€€€€€€ 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€≈
D__inference_dense_9_layer_call_and_return_conditional_losses_1312809}%&<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€€€€€€€€€€
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ Я
)__inference_dense_9_layer_call_fn_1312778r%&<Ґ9
2Ґ/
-К*
inputs€€€€€€€€€€€€€€€€€€
™ ".К+
unknown€€€€€€€€€€€€€€€€€€„
A__inference_lstm_layer_call_and_return_conditional_losses_1311448С/01OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€


 
p

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€ 
Ъ „
A__inference_lstm_layer_call_and_return_conditional_losses_1311875С/01OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€


 
p 

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€ 
Ъ –
A__inference_lstm_layer_call_and_return_conditional_losses_1312302К/01HҐE
>Ґ;
-К*
inputs€€€€€€€€€€€€€€€€€€


 
p

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€ 
Ъ –
A__inference_lstm_layer_call_and_return_conditional_losses_1312729К/01HҐE
>Ґ;
-К*
inputs€€€€€€€€€€€€€€€€€€


 
p 

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€ 
Ъ ±
&__inference_lstm_layer_call_fn_1310988Ж/01OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€


 
p

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€ ±
&__inference_lstm_layer_call_fn_1310999Ж/01OҐL
EҐB
4Ъ1
/К,
inputs_0€€€€€€€€€€€€€€€€€€


 
p 

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€ ©
&__inference_lstm_layer_call_fn_1311010/01HҐE
>Ґ;
-К*
inputs€€€€€€€€€€€€€€€€€€


 
p

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€ ©
&__inference_lstm_layer_call_fn_1311021/01HҐE
>Ґ;
-К*
inputs€€€€€€€€€€€€€€€€€€


 
p 

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€ ё
I__inference_sequential_1_layer_call_and_return_conditional_losses_1310403Р	/01%&-.HҐE
>Ґ;
1К.

lstm_input€€€€€€€€€€€€€€€€€€

p

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ ё
I__inference_sequential_1_layer_call_and_return_conditional_losses_1310856Р	/01%&-.HҐE
>Ґ;
1К.

lstm_input€€€€€€€€€€€€€€€€€€

p 

 
™ "9Ґ6
/К,
tensor_0€€€€€€€€€€€€€€€€€€
Ъ Є
.__inference_sequential_1_layer_call_fn_1310879Е	/01%&-.HҐE
>Ґ;
1К.

lstm_input€€€€€€€€€€€€€€€€€€

p

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€Є
.__inference_sequential_1_layer_call_fn_1310902Е	/01%&-.HҐE
>Ґ;
1К.

lstm_input€€€€€€€€€€€€€€€€€€

p 

 
™ ".К+
unknown€€€€€€€€€€€€€€€€€€«
%__inference_signature_wrapper_1310977Э	/01%&-.NҐK
Ґ 
D™A
?

lstm_input1К.

lstm_input€€€€€€€€€€€€€€€€€€
"@™=
;
dense_10/К,
dense_10€€€€€€€€€€€€€€€€€€