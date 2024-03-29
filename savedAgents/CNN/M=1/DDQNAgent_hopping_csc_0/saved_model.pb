��

��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
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
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12unknown8��
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:  *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: d*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: d*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:d*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:d*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
�
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/m
�
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
: *
dtype0
�
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv2d_1/kernel/m
�
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:  *
dtype0
�
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
: *
dtype0
�
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: d*$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

: d*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:d*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:d*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d/kernel/v
�
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
: *
dtype0
�
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *'
shared_nameAdam/conv2d_1/kernel/v
�
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:  *
dtype0
�
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
: *
dtype0
�
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: d*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

: d*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:d*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:d*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�-
value�-B�- B�-
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
�
(iter

)beta_1

*beta_2
	+decay
,learning_ratemPmQmRmSmTmU"mV#mWvXvYvZv[v\v]"v^#v_
8
0
1
2
3
4
5
"6
#7
8
0
1
2
3
4
5
"6
#7
 
�
-layer_regularization_losses
.metrics
/layer_metrics
	variables
trainable_variables

0layers
1non_trainable_variables
	regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
2layer_regularization_losses
3metrics
4layer_metrics
	variables
trainable_variables

5layers
regularization_losses
6non_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
7layer_regularization_losses
8metrics
9layer_metrics
	variables
trainable_variables

:layers
regularization_losses
;non_trainable_variables
 
 
 
�
<layer_regularization_losses
=metrics
>layer_metrics
	variables
trainable_variables

?layers
regularization_losses
@non_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
Alayer_regularization_losses
Bmetrics
Clayer_metrics
	variables
trainable_variables

Dlayers
 regularization_losses
Enon_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
�
Flayer_regularization_losses
Gmetrics
Hlayer_metrics
$	variables
%trainable_variables

Ilayers
&regularization_losses
Jnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

K0
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ltotal
	Mcount
N	variables
O	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

N	variables
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_conv2d_inputPlaceholder*3
_output_shapes!
:���������*
dtype0*(
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference_signature_wrapper_3103246
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *)
f$R"
 __inference__traced_save_3103695
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__traced_restore_3103798��
�	
�
D__inference_dense_1_layer_call_and_return_conditional_losses_3103080

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_3103034

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:��������� 2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� :[ W
3
_output_shapes!
:��������� 
 
_user_specified_nameinputs
�*
�
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3103012

inputs)
%conv2d_conv2d_readvariableop_resource6
2squeeze_batch_dims_biasadd_readvariableop_resource
identity��Conv2D/Conv2D/ReadVariableOp�)squeeze_batch_dims/BiasAdd/ReadVariableOpR
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:2
Conv2D/Shape�
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv2D/strided_slice/stack�
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
Conv2D/strided_slice/stack_1�
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Conv2D/strided_slice/stack_2�
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
Conv2D/strided_slice�
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2
Conv2D/Reshape/shape�
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2
Conv2D/Reshape�
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/Conv2D/ReadVariableOp�
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Conv2D/Conv2D�
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2
Conv2D/concat/values_1s
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Conv2D/concat/axis�
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Conv2D/concat�
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:��������� 2
Conv2D/Reshape_1}
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape�
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stack�
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2*
(squeeze_batch_dims/strided_slice/stack_1�
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2�
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice�
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2"
 squeeze_batch_dims/Reshape/shape�
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2
squeeze_batch_dims/Reshape�
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOp�
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
squeeze_batch_dims/BiasAdd�
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2$
"squeeze_batch_dims/concat/values_1�
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
squeeze_batch_dims/concat/axis�
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concat�
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:��������� 2
squeeze_batch_dims/Reshape_1y
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:��������� 2
Relu�
IdentityIdentityRelu:activations:0^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*
T0*3
_output_shapes!
:��������� 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':��������� ::2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:��������� 
 
_user_specified_nameinputs
�
}
(__inference_conv2d_layer_call_fn_3103486

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_31029632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:��������� 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�

*__inference_conv2d_1_layer_call_fn_3103528

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_31030122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*3
_output_shapes!
:��������� 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:��������� 
 
_user_specified_nameinputs
�q
�
G__inference_sequential_layer_call_and_return_conditional_losses_3103402

inputs0
,conv2d_conv2d_conv2d_readvariableop_resource=
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource2
.conv2d_1_conv2d_conv2d_readvariableop_resource?
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��#conv2d/Conv2D/Conv2D/ReadVariableOp�0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp�%conv2d_1/Conv2D/Conv2D/ReadVariableOp�2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp`
conv2d/Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d/Conv2D/Shape�
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv2d/Conv2D/strided_slice/stack�
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2%
#conv2d/Conv2D/strided_slice/stack_1�
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv2d/Conv2D/strided_slice/stack_2�
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv2d/Conv2D/strided_slice�
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         2
conv2d/Conv2D/Reshape/shape�
conv2d/Conv2D/ReshapeReshapeinputs$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
conv2d/Conv2D/Reshape�
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02%
#conv2d/Conv2D/Conv2D/ReadVariableOp�
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d/Conv2D/Conv2D�
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2
conv2d/Conv2D/concat/values_1�
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv2d/Conv2D/concat/axis�
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv2d/Conv2D/concat�
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:��������� 2
conv2d/Conv2D/Reshape_1�
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:2!
conv2d/squeeze_batch_dims/Shape�
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv2d/squeeze_batch_dims/strided_slice/stack�
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������21
/conv2d/squeeze_batch_dims/strided_slice/stack_1�
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv2d/squeeze_batch_dims/strided_slice/stack_2�
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv2d/squeeze_batch_dims/strided_slice�
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2)
'conv2d/squeeze_batch_dims/Reshape/shape�
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2#
!conv2d/squeeze_batch_dims/Reshape�
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp�
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2#
!conv2d/squeeze_batch_dims/BiasAdd�
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2+
)conv2d/squeeze_batch_dims/concat/values_1�
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%conv2d/squeeze_batch_dims/concat/axis�
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv2d/squeeze_batch_dims/concat�
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:��������� 2%
#conv2d/squeeze_batch_dims/Reshape_1�
conv2d/ReluRelu,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:��������� 2
conv2d/Reluw
conv2d_1/Conv2D/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_1/Conv2D/Shape�
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#conv2d_1/Conv2D/strided_slice/stack�
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2'
%conv2d_1/Conv2D/strided_slice/stack_1�
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%conv2d_1/Conv2D/strided_slice/stack_2�
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv2d_1/Conv2D/strided_slice�
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2
conv2d_1/Conv2D/Reshape/shape�
conv2d_1/Conv2D/ReshapeReshapeconv2d/Relu:activations:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_1/Conv2D/Reshape�
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02'
%conv2d_1/Conv2D/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d_1/Conv2D/Conv2D�
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2!
conv2d_1/Conv2D/concat/values_1�
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv2d_1/Conv2D/concat/axis�
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv2d_1/Conv2D/concat�
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:��������� 2
conv2d_1/Conv2D/Reshape_1�
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:2#
!conv2d_1/squeeze_batch_dims/Shape�
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/conv2d_1/squeeze_batch_dims/strided_slice/stack�
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������23
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1�
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2�
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2+
)conv2d_1/squeeze_batch_dims/strided_slice�
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2+
)conv2d_1/squeeze_batch_dims/Reshape/shape�
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2%
#conv2d_1/squeeze_batch_dims/Reshape�
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp�
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2%
#conv2d_1/squeeze_batch_dims/BiasAdd�
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2-
+conv2d_1/squeeze_batch_dims/concat/values_1�
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'conv2d_1/squeeze_batch_dims/concat/axis�
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"conv2d_1/squeeze_batch_dims/concat�
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:��������� 2'
%conv2d_1/squeeze_batch_dims/Reshape_1�
conv2d_1/ReluRelu.conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:��������� 2
conv2d_1/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
flatten/Const�
flatten/ReshapeReshapeconv2d_1/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:��������� 2
flatten/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: d*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Softmax�
IdentityIdentitydense_1/Softmax:softmax:0$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�q
�
G__inference_sequential_layer_call_and_return_conditional_losses_3103324

inputs0
,conv2d_conv2d_conv2d_readvariableop_resource=
9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource2
.conv2d_1_conv2d_conv2d_readvariableop_resource?
;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity��#conv2d/Conv2D/Conv2D/ReadVariableOp�0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp�%conv2d_1/Conv2D/Conv2D/ReadVariableOp�2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp`
conv2d/Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:2
conv2d/Conv2D/Shape�
!conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!conv2d/Conv2D/strided_slice/stack�
#conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2%
#conv2d/Conv2D/strided_slice/stack_1�
#conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#conv2d/Conv2D/strided_slice/stack_2�
conv2d/Conv2D/strided_sliceStridedSliceconv2d/Conv2D/Shape:output:0*conv2d/Conv2D/strided_slice/stack:output:0,conv2d/Conv2D/strided_slice/stack_1:output:0,conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv2d/Conv2D/strided_slice�
conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         2
conv2d/Conv2D/Reshape/shape�
conv2d/Conv2D/ReshapeReshapeinputs$conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
conv2d/Conv2D/Reshape�
#conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp,conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02%
#conv2d/Conv2D/Conv2D/ReadVariableOp�
conv2d/Conv2D/Conv2DConv2Dconv2d/Conv2D/Reshape:output:0+conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d/Conv2D/Conv2D�
conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2
conv2d/Conv2D/concat/values_1�
conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv2d/Conv2D/concat/axis�
conv2d/Conv2D/concatConcatV2$conv2d/Conv2D/strided_slice:output:0&conv2d/Conv2D/concat/values_1:output:0"conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv2d/Conv2D/concat�
conv2d/Conv2D/Reshape_1Reshapeconv2d/Conv2D/Conv2D:output:0conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:��������� 2
conv2d/Conv2D/Reshape_1�
conv2d/squeeze_batch_dims/ShapeShape conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:2!
conv2d/squeeze_batch_dims/Shape�
-conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-conv2d/squeeze_batch_dims/strided_slice/stack�
/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������21
/conv2d/squeeze_batch_dims/strided_slice/stack_1�
/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/conv2d/squeeze_batch_dims/strided_slice/stack_2�
'conv2d/squeeze_batch_dims/strided_sliceStridedSlice(conv2d/squeeze_batch_dims/Shape:output:06conv2d/squeeze_batch_dims/strided_slice/stack:output:08conv2d/squeeze_batch_dims/strided_slice/stack_1:output:08conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'conv2d/squeeze_batch_dims/strided_slice�
'conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2)
'conv2d/squeeze_batch_dims/Reshape/shape�
!conv2d/squeeze_batch_dims/ReshapeReshape conv2d/Conv2D/Reshape_1:output:00conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2#
!conv2d/squeeze_batch_dims/Reshape�
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp9conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp�
!conv2d/squeeze_batch_dims/BiasAddBiasAdd*conv2d/squeeze_batch_dims/Reshape:output:08conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2#
!conv2d/squeeze_batch_dims/BiasAdd�
)conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2+
)conv2d/squeeze_batch_dims/concat/values_1�
%conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2'
%conv2d/squeeze_batch_dims/concat/axis�
 conv2d/squeeze_batch_dims/concatConcatV20conv2d/squeeze_batch_dims/strided_slice:output:02conv2d/squeeze_batch_dims/concat/values_1:output:0.conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 conv2d/squeeze_batch_dims/concat�
#conv2d/squeeze_batch_dims/Reshape_1Reshape*conv2d/squeeze_batch_dims/BiasAdd:output:0)conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:��������� 2%
#conv2d/squeeze_batch_dims/Reshape_1�
conv2d/ReluRelu,conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:��������� 2
conv2d/Reluw
conv2d_1/Conv2D/ShapeShapeconv2d/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_1/Conv2D/Shape�
#conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#conv2d_1/Conv2D/strided_slice/stack�
%conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2'
%conv2d_1/Conv2D/strided_slice/stack_1�
%conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%conv2d_1/Conv2D/strided_slice/stack_2�
conv2d_1/Conv2D/strided_sliceStridedSliceconv2d_1/Conv2D/Shape:output:0,conv2d_1/Conv2D/strided_slice/stack:output:0.conv2d_1/Conv2D/strided_slice/stack_1:output:0.conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
conv2d_1/Conv2D/strided_slice�
conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2
conv2d_1/Conv2D/Reshape/shape�
conv2d_1/Conv2D/ReshapeReshapeconv2d/Relu:activations:0&conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2
conv2d_1/Conv2D/Reshape�
%conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02'
%conv2d_1/Conv2D/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/Conv2DConv2D conv2d_1/Conv2D/Reshape:output:0-conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
conv2d_1/Conv2D/Conv2D�
conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2!
conv2d_1/Conv2D/concat/values_1�
conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
conv2d_1/Conv2D/concat/axis�
conv2d_1/Conv2D/concatConcatV2&conv2d_1/Conv2D/strided_slice:output:0(conv2d_1/Conv2D/concat/values_1:output:0$conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv2d_1/Conv2D/concat�
conv2d_1/Conv2D/Reshape_1Reshapeconv2d_1/Conv2D/Conv2D:output:0conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:��������� 2
conv2d_1/Conv2D/Reshape_1�
!conv2d_1/squeeze_batch_dims/ShapeShape"conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:2#
!conv2d_1/squeeze_batch_dims/Shape�
/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/conv2d_1/squeeze_batch_dims/strided_slice/stack�
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������23
1conv2d_1/squeeze_batch_dims/strided_slice/stack_1�
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1conv2d_1/squeeze_batch_dims/strided_slice/stack_2�
)conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice*conv2d_1/squeeze_batch_dims/Shape:output:08conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0:conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2+
)conv2d_1/squeeze_batch_dims/strided_slice�
)conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2+
)conv2d_1/squeeze_batch_dims/Reshape/shape�
#conv2d_1/squeeze_batch_dims/ReshapeReshape"conv2d_1/Conv2D/Reshape_1:output:02conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2%
#conv2d_1/squeeze_batch_dims/Reshape�
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp;conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype024
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp�
#conv2d_1/squeeze_batch_dims/BiasAddBiasAdd,conv2d_1/squeeze_batch_dims/Reshape:output:0:conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2%
#conv2d_1/squeeze_batch_dims/BiasAdd�
+conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2-
+conv2d_1/squeeze_batch_dims/concat/values_1�
'conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2)
'conv2d_1/squeeze_batch_dims/concat/axis�
"conv2d_1/squeeze_batch_dims/concatConcatV22conv2d_1/squeeze_batch_dims/strided_slice:output:04conv2d_1/squeeze_batch_dims/concat/values_1:output:00conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2$
"conv2d_1/squeeze_batch_dims/concat�
%conv2d_1/squeeze_batch_dims/Reshape_1Reshape,conv2d_1/squeeze_batch_dims/BiasAdd:output:0+conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:��������� 2'
%conv2d_1/squeeze_batch_dims/Reshape_1�
conv2d_1/ReluRelu.conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:��������� 2
conv2d_1/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
flatten/Const�
flatten/ReshapeReshapeconv2d_1/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:��������� 2
flatten/Reshape�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: d*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
dense_1/Softmax�
IdentityIdentitydense_1/Softmax:softmax:0$^conv2d/Conv2D/Conv2D/ReadVariableOp1^conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp&^conv2d_1/Conv2D/Conv2D/ReadVariableOp3^conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::2J
#conv2d/Conv2D/Conv2D/ReadVariableOp#conv2d/Conv2D/Conv2D/ReadVariableOp2d
0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp0conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2N
%conv2d_1/Conv2D/Conv2D/ReadVariableOp%conv2d_1/Conv2D/Conv2D/ReadVariableOp2h
2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�	
�
B__inference_dense_layer_call_and_return_conditional_losses_3103550

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
~
)__inference_dense_1_layer_call_fn_3103579

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_31030802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
|
'__inference_dense_layer_call_fn_3103559

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_31030532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
G__inference_sequential_layer_call_and_return_conditional_losses_3103122
conv2d_input
conv2d_3103100
conv2d_3103102
conv2d_1_3103105
conv2d_1_3103107
dense_3103111
dense_3103113
dense_1_3103116
dense_1_3103118
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_3103100conv2d_3103102*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_31029632 
conv2d/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_3103105conv2d_1_3103107*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_31030122"
 conv2d_1/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_31030342
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3103111dense_3103113*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_31030532
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3103116dense_1_3103118*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_31030802!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:a ]
3
_output_shapes!
:���������
&
_user_specified_nameconv2d_input
ʃ
�
#__inference__traced_restore_3103798
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count,
(assignvariableop_15_adam_conv2d_kernel_m*
&assignvariableop_16_adam_conv2d_bias_m.
*assignvariableop_17_adam_conv2d_1_kernel_m,
(assignvariableop_18_adam_conv2d_1_bias_m+
'assignvariableop_19_adam_dense_kernel_m)
%assignvariableop_20_adam_dense_bias_m-
)assignvariableop_21_adam_dense_1_kernel_m+
'assignvariableop_22_adam_dense_1_bias_m,
(assignvariableop_23_adam_conv2d_kernel_v*
&assignvariableop_24_adam_conv2d_bias_v.
*assignvariableop_25_adam_conv2d_1_kernel_v,
(assignvariableop_26_adam_conv2d_1_bias_v+
'assignvariableop_27_adam_dense_kernel_v)
%assignvariableop_28_adam_dense_bias_v-
)assignvariableop_29_adam_dense_1_kernel_v+
'assignvariableop_30_adam_dense_1_bias_v
identity_32��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp(assignvariableop_15_adam_conv2d_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_conv2d_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_conv2d_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_conv2d_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_dense_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp%assignvariableop_20_adam_dense_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_dense_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_dense_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_conv2d_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_conv2d_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31�
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*�
_input_shapes�
~: :::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
G__inference_sequential_layer_call_and_return_conditional_losses_3103097
conv2d_input
conv2d_3102974
conv2d_3102976
conv2d_1_3103023
conv2d_1_3103025
dense_3103064
dense_3103066
dense_1_3103091
dense_1_3103093
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_3102974conv2d_3102976*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_31029632 
conv2d/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_3103023conv2d_1_3103025*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_31030122"
 conv2d_1/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_31030342
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3103064dense_3103066*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_31030532
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3103091dense_1_3103093*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_31030802!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:a ]
3
_output_shapes!
:���������
&
_user_specified_nameconv2d_input
�
�
%__inference_signature_wrapper_3103246
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__wrapped_model_31029262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
3
_output_shapes!
:���������
&
_user_specified_nameconv2d_input
�
�
,__inference_sequential_layer_call_fn_3103169
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_31031502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
3
_output_shapes!
:���������
&
_user_specified_nameconv2d_input
�
`
D__inference_flatten_layer_call_and_return_conditional_losses_3103534

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:��������� 2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� :[ W
3
_output_shapes!
:��������� 
 
_user_specified_nameinputs
�	
�
B__inference_dense_layer_call_and_return_conditional_losses_3103053

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������d2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������d2

Identity"
identityIdentity:output:0*.
_input_shapes
:��������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�*
�
C__inference_conv2d_layer_call_and_return_conditional_losses_3103477

inputs)
%conv2d_conv2d_readvariableop_resource6
2squeeze_batch_dims_biasadd_readvariableop_resource
identity��Conv2D/Conv2D/ReadVariableOp�)squeeze_batch_dims/BiasAdd/ReadVariableOpR
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:2
Conv2D/Shape�
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv2D/strided_slice/stack�
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
Conv2D/strided_slice/stack_1�
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Conv2D/strided_slice/stack_2�
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
Conv2D/strided_slice�
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         2
Conv2D/Reshape/shape�
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
Conv2D/Reshape�
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/Conv2D/ReadVariableOp�
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Conv2D/Conv2D�
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2
Conv2D/concat/values_1s
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Conv2D/concat/axis�
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Conv2D/concat�
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:��������� 2
Conv2D/Reshape_1}
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape�
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stack�
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2*
(squeeze_batch_dims/strided_slice/stack_1�
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2�
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice�
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2"
 squeeze_batch_dims/Reshape/shape�
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2
squeeze_batch_dims/Reshape�
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOp�
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
squeeze_batch_dims/BiasAdd�
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2$
"squeeze_batch_dims/concat/values_1�
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
squeeze_batch_dims/concat/axis�
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concat�
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:��������� 2
squeeze_batch_dims/Reshape_1y
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:��������� 2
Relu�
IdentityIdentityRelu:activations:0^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*
T0*3
_output_shapes!
:��������� 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������::2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�*
�
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3103519

inputs)
%conv2d_conv2d_readvariableop_resource6
2squeeze_batch_dims_biasadd_readvariableop_resource
identity��Conv2D/Conv2D/ReadVariableOp�)squeeze_batch_dims/BiasAdd/ReadVariableOpR
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:2
Conv2D/Shape�
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv2D/strided_slice/stack�
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
Conv2D/strided_slice/stack_1�
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Conv2D/strided_slice/stack_2�
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
Conv2D/strided_slice�
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2
Conv2D/Reshape/shape�
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2
Conv2D/Reshape�
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/Conv2D/ReadVariableOp�
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Conv2D/Conv2D�
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2
Conv2D/concat/values_1s
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Conv2D/concat/axis�
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Conv2D/concat�
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:��������� 2
Conv2D/Reshape_1}
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape�
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stack�
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2*
(squeeze_batch_dims/strided_slice/stack_1�
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2�
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice�
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2"
 squeeze_batch_dims/Reshape/shape�
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2
squeeze_batch_dims/Reshape�
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOp�
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
squeeze_batch_dims/BiasAdd�
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2$
"squeeze_batch_dims/concat/values_1�
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
squeeze_batch_dims/concat/axis�
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concat�
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:��������� 2
squeeze_batch_dims/Reshape_1y
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:��������� 2
Relu�
IdentityIdentityRelu:activations:0^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*
T0*3
_output_shapes!
:��������� 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':��������� ::2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:��������� 
 
_user_specified_nameinputs
�
E
)__inference_flatten_layer_call_fn_3103539

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_31030342
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:��������� 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:��������� :[ W
3
_output_shapes!
:��������� 
 
_user_specified_nameinputs
�
�
,__inference_sequential_layer_call_fn_3103423

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_31031502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
,__inference_sequential_layer_call_fn_3103215
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_31031962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
3
_output_shapes!
:���������
&
_user_specified_nameconv2d_input
�
�
,__inference_sequential_layer_call_fn_3103444

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_31031962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�
�
G__inference_sequential_layer_call_and_return_conditional_losses_3103196

inputs
conv2d_3103174
conv2d_3103176
conv2d_1_3103179
conv2d_1_3103181
dense_3103185
dense_3103187
dense_1_3103190
dense_1_3103192
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3103174conv2d_3103176*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_31029632 
conv2d/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_3103179conv2d_1_3103181*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_31030122"
 conv2d_1/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_31030342
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3103185dense_3103187*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_31030532
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3103190dense_1_3103192*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_31030802!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�E
�
 __inference__traced_save_3103695
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : :  : : d:d:d:: : : : : : : : : :  : : d:d:d:: : :  : : d:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :$ 

_output_shapes

: d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :$ 

_output_shapes

: d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :$ 

_output_shapes

: d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
:: 

_output_shapes
: 
�
�
G__inference_sequential_layer_call_and_return_conditional_losses_3103150

inputs
conv2d_3103128
conv2d_3103130
conv2d_1_3103133
conv2d_1_3103135
dense_3103139
dense_3103141
dense_1_3103144
dense_1_3103146
identity��conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3103128conv2d_3103130*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_31029632 
conv2d/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_3103133conv2d_1_3103135*
Tin
2*
Tout
2*
_collective_manager_ids
 *3
_output_shapes!
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_31030122"
 conv2d_1/StatefulPartitionedCall�
flatten/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_31030342
flatten/PartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_3103139dense_3103141*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_31030532
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_3103144dense_1_3103146*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_31030802!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
�	
�
D__inference_dense_1_layer_call_and_return_conditional_losses_3103570

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������d::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�*
�
C__inference_conv2d_layer_call_and_return_conditional_losses_3102963

inputs)
%conv2d_conv2d_readvariableop_resource6
2squeeze_batch_dims_biasadd_readvariableop_resource
identity��Conv2D/Conv2D/ReadVariableOp�)squeeze_batch_dims/BiasAdd/ReadVariableOpR
Conv2D/ShapeShapeinputs*
T0*
_output_shapes
:2
Conv2D/Shape�
Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Conv2D/strided_slice/stack�
Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2
Conv2D/strided_slice/stack_1�
Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Conv2D/strided_slice/stack_2�
Conv2D/strided_sliceStridedSliceConv2D/Shape:output:0#Conv2D/strided_slice/stack:output:0%Conv2D/strided_slice/stack_1:output:0%Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
Conv2D/strided_slice�
Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         2
Conv2D/Reshape/shape�
Conv2D/ReshapeReshapeinputsConv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2
Conv2D/Reshape�
Conv2D/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/Conv2D/ReadVariableOp�
Conv2D/Conv2DConv2DConv2D/Reshape:output:0$Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2
Conv2D/Conv2D�
Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2
Conv2D/concat/values_1s
Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Conv2D/concat/axis�
Conv2D/concatConcatV2Conv2D/strided_slice:output:0Conv2D/concat/values_1:output:0Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Conv2D/concat�
Conv2D/Reshape_1ReshapeConv2D/Conv2D:output:0Conv2D/concat:output:0*
T0*3
_output_shapes!
:��������� 2
Conv2D/Reshape_1}
squeeze_batch_dims/ShapeShapeConv2D/Reshape_1:output:0*
T0*
_output_shapes
:2
squeeze_batch_dims/Shape�
&squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&squeeze_batch_dims/strided_slice/stack�
(squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2*
(squeeze_batch_dims/strided_slice/stack_1�
(squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(squeeze_batch_dims/strided_slice/stack_2�
 squeeze_batch_dims/strided_sliceStridedSlice!squeeze_batch_dims/Shape:output:0/squeeze_batch_dims/strided_slice/stack:output:01squeeze_batch_dims/strided_slice/stack_1:output:01squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2"
 squeeze_batch_dims/strided_slice�
 squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2"
 squeeze_batch_dims/Reshape/shape�
squeeze_batch_dims/ReshapeReshapeConv2D/Reshape_1:output:0)squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2
squeeze_batch_dims/Reshape�
)squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOp2squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)squeeze_batch_dims/BiasAdd/ReadVariableOp�
squeeze_batch_dims/BiasAddBiasAdd#squeeze_batch_dims/Reshape:output:01squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2
squeeze_batch_dims/BiasAdd�
"squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2$
"squeeze_batch_dims/concat/values_1�
squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2 
squeeze_batch_dims/concat/axis�
squeeze_batch_dims/concatConcatV2)squeeze_batch_dims/strided_slice:output:0+squeeze_batch_dims/concat/values_1:output:0'squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2
squeeze_batch_dims/concat�
squeeze_batch_dims/Reshape_1Reshape#squeeze_batch_dims/BiasAdd:output:0"squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:��������� 2
squeeze_batch_dims/Reshape_1y
ReluRelu%squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:��������� 2
Relu�
IdentityIdentityRelu:activations:0^Conv2D/Conv2D/ReadVariableOp*^squeeze_batch_dims/BiasAdd/ReadVariableOp*
T0*3
_output_shapes!
:��������� 2

Identity"
identityIdentity:output:0*:
_input_shapes)
':���������::2<
Conv2D/Conv2D/ReadVariableOpConv2D/Conv2D/ReadVariableOp2V
)squeeze_batch_dims/BiasAdd/ReadVariableOp)squeeze_batch_dims/BiasAdd/ReadVariableOp:[ W
3
_output_shapes!
:���������
 
_user_specified_nameinputs
��
�
"__inference__wrapped_model_3102926
conv2d_input;
7sequential_conv2d_conv2d_conv2d_readvariableop_resourceH
Dsequential_conv2d_squeeze_batch_dims_biasadd_readvariableop_resource=
9sequential_conv2d_1_conv2d_conv2d_readvariableop_resourceJ
Fsequential_conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identity��.sequential/conv2d/Conv2D/Conv2D/ReadVariableOp�;sequential/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp�0sequential/conv2d_1/Conv2D/Conv2D/ReadVariableOp�=sequential/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp�'sequential/dense/BiasAdd/ReadVariableOp�&sequential/dense/MatMul/ReadVariableOp�)sequential/dense_1/BiasAdd/ReadVariableOp�(sequential/dense_1/MatMul/ReadVariableOp|
sequential/conv2d/Conv2D/ShapeShapeconv2d_input*
T0*
_output_shapes
:2 
sequential/conv2d/Conv2D/Shape�
,sequential/conv2d/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential/conv2d/Conv2D/strided_slice/stack�
.sequential/conv2d/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������20
.sequential/conv2d/Conv2D/strided_slice/stack_1�
.sequential/conv2d/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.sequential/conv2d/Conv2D/strided_slice/stack_2�
&sequential/conv2d/Conv2D/strided_sliceStridedSlice'sequential/conv2d/Conv2D/Shape:output:05sequential/conv2d/Conv2D/strided_slice/stack:output:07sequential/conv2d/Conv2D/strided_slice/stack_1:output:07sequential/conv2d/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2(
&sequential/conv2d/Conv2D/strided_slice�
&sequential/conv2d/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����         2(
&sequential/conv2d/Conv2D/Reshape/shape�
 sequential/conv2d/Conv2D/ReshapeReshapeconv2d_input/sequential/conv2d/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:���������2"
 sequential/conv2d/Conv2D/Reshape�
.sequential/conv2d/Conv2D/Conv2D/ReadVariableOpReadVariableOp7sequential_conv2d_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.sequential/conv2d/Conv2D/Conv2D/ReadVariableOp�
sequential/conv2d/Conv2D/Conv2DConv2D)sequential/conv2d/Conv2D/Reshape:output:06sequential/conv2d/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2!
sequential/conv2d/Conv2D/Conv2D�
(sequential/conv2d/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2*
(sequential/conv2d/Conv2D/concat/values_1�
$sequential/conv2d/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2&
$sequential/conv2d/Conv2D/concat/axis�
sequential/conv2d/Conv2D/concatConcatV2/sequential/conv2d/Conv2D/strided_slice:output:01sequential/conv2d/Conv2D/concat/values_1:output:0-sequential/conv2d/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:2!
sequential/conv2d/Conv2D/concat�
"sequential/conv2d/Conv2D/Reshape_1Reshape(sequential/conv2d/Conv2D/Conv2D:output:0(sequential/conv2d/Conv2D/concat:output:0*
T0*3
_output_shapes!
:��������� 2$
"sequential/conv2d/Conv2D/Reshape_1�
*sequential/conv2d/squeeze_batch_dims/ShapeShape+sequential/conv2d/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:2,
*sequential/conv2d/squeeze_batch_dims/Shape�
8sequential/conv2d/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential/conv2d/squeeze_batch_dims/strided_slice/stack�
:sequential/conv2d/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2<
:sequential/conv2d/squeeze_batch_dims/strided_slice/stack_1�
:sequential/conv2d/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential/conv2d/squeeze_batch_dims/strided_slice/stack_2�
2sequential/conv2d/squeeze_batch_dims/strided_sliceStridedSlice3sequential/conv2d/squeeze_batch_dims/Shape:output:0Asequential/conv2d/squeeze_batch_dims/strided_slice/stack:output:0Csequential/conv2d/squeeze_batch_dims/strided_slice/stack_1:output:0Csequential/conv2d/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask24
2sequential/conv2d/squeeze_batch_dims/strided_slice�
2sequential/conv2d/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          24
2sequential/conv2d/squeeze_batch_dims/Reshape/shape�
,sequential/conv2d/squeeze_batch_dims/ReshapeReshape+sequential/conv2d/Conv2D/Reshape_1:output:0;sequential/conv2d/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2.
,sequential/conv2d/squeeze_batch_dims/Reshape�
;sequential/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpDsequential_conv2d_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02=
;sequential/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp�
,sequential/conv2d/squeeze_batch_dims/BiasAddBiasAdd5sequential/conv2d/squeeze_batch_dims/Reshape:output:0Csequential/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 2.
,sequential/conv2d/squeeze_batch_dims/BiasAdd�
4sequential/conv2d/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          26
4sequential/conv2d/squeeze_batch_dims/concat/values_1�
0sequential/conv2d/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������22
0sequential/conv2d/squeeze_batch_dims/concat/axis�
+sequential/conv2d/squeeze_batch_dims/concatConcatV2;sequential/conv2d/squeeze_batch_dims/strided_slice:output:0=sequential/conv2d/squeeze_batch_dims/concat/values_1:output:09sequential/conv2d/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2-
+sequential/conv2d/squeeze_batch_dims/concat�
.sequential/conv2d/squeeze_batch_dims/Reshape_1Reshape5sequential/conv2d/squeeze_batch_dims/BiasAdd:output:04sequential/conv2d/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:��������� 20
.sequential/conv2d/squeeze_batch_dims/Reshape_1�
sequential/conv2d/ReluRelu7sequential/conv2d/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:��������� 2
sequential/conv2d/Relu�
 sequential/conv2d_1/Conv2D/ShapeShape$sequential/conv2d/Relu:activations:0*
T0*
_output_shapes
:2"
 sequential/conv2d_1/Conv2D/Shape�
.sequential/conv2d_1/Conv2D/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.sequential/conv2d_1/Conv2D/strided_slice/stack�
0sequential/conv2d_1/Conv2D/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������22
0sequential/conv2d_1/Conv2D/strided_slice/stack_1�
0sequential/conv2d_1/Conv2D/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential/conv2d_1/Conv2D/strided_slice/stack_2�
(sequential/conv2d_1/Conv2D/strided_sliceStridedSlice)sequential/conv2d_1/Conv2D/Shape:output:07sequential/conv2d_1/Conv2D/strided_slice/stack:output:09sequential/conv2d_1/Conv2D/strided_slice/stack_1:output:09sequential/conv2d_1/Conv2D/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2*
(sequential/conv2d_1/Conv2D/strided_slice�
(sequential/conv2d_1/Conv2D/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          2*
(sequential/conv2d_1/Conv2D/Reshape/shape�
"sequential/conv2d_1/Conv2D/ReshapeReshape$sequential/conv2d/Relu:activations:01sequential/conv2d_1/Conv2D/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 2$
"sequential/conv2d_1/Conv2D/Reshape�
0sequential/conv2d_1/Conv2D/Conv2D/ReadVariableOpReadVariableOp9sequential_conv2d_1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype022
0sequential/conv2d_1/Conv2D/Conv2D/ReadVariableOp�
!sequential/conv2d_1/Conv2D/Conv2DConv2D+sequential/conv2d_1/Conv2D/Reshape:output:08sequential/conv2d_1/Conv2D/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� *
paddingVALID*
strides
2#
!sequential/conv2d_1/Conv2D/Conv2D�
*sequential/conv2d_1/Conv2D/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          2,
*sequential/conv2d_1/Conv2D/concat/values_1�
&sequential/conv2d_1/Conv2D/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2(
&sequential/conv2d_1/Conv2D/concat/axis�
!sequential/conv2d_1/Conv2D/concatConcatV21sequential/conv2d_1/Conv2D/strided_slice:output:03sequential/conv2d_1/Conv2D/concat/values_1:output:0/sequential/conv2d_1/Conv2D/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!sequential/conv2d_1/Conv2D/concat�
$sequential/conv2d_1/Conv2D/Reshape_1Reshape*sequential/conv2d_1/Conv2D/Conv2D:output:0*sequential/conv2d_1/Conv2D/concat:output:0*
T0*3
_output_shapes!
:��������� 2&
$sequential/conv2d_1/Conv2D/Reshape_1�
,sequential/conv2d_1/squeeze_batch_dims/ShapeShape-sequential/conv2d_1/Conv2D/Reshape_1:output:0*
T0*
_output_shapes
:2.
,sequential/conv2d_1/squeeze_batch_dims/Shape�
:sequential/conv2d_1/squeeze_batch_dims/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:sequential/conv2d_1/squeeze_batch_dims/strided_slice/stack�
<sequential/conv2d_1/squeeze_batch_dims/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������2>
<sequential/conv2d_1/squeeze_batch_dims/strided_slice/stack_1�
<sequential/conv2d_1/squeeze_batch_dims/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential/conv2d_1/squeeze_batch_dims/strided_slice/stack_2�
4sequential/conv2d_1/squeeze_batch_dims/strided_sliceStridedSlice5sequential/conv2d_1/squeeze_batch_dims/Shape:output:0Csequential/conv2d_1/squeeze_batch_dims/strided_slice/stack:output:0Esequential/conv2d_1/squeeze_batch_dims/strided_slice/stack_1:output:0Esequential/conv2d_1/squeeze_batch_dims/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask26
4sequential/conv2d_1/squeeze_batch_dims/strided_slice�
4sequential/conv2d_1/squeeze_batch_dims/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"����          26
4sequential/conv2d_1/squeeze_batch_dims/Reshape/shape�
.sequential/conv2d_1/squeeze_batch_dims/ReshapeReshape-sequential/conv2d_1/Conv2D/Reshape_1:output:0=sequential/conv2d_1/squeeze_batch_dims/Reshape/shape:output:0*
T0*/
_output_shapes
:��������� 20
.sequential/conv2d_1/squeeze_batch_dims/Reshape�
=sequential/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOpReadVariableOpFsequential_conv2d_1_squeeze_batch_dims_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02?
=sequential/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp�
.sequential/conv2d_1/squeeze_batch_dims/BiasAddBiasAdd7sequential/conv2d_1/squeeze_batch_dims/Reshape:output:0Esequential/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:��������� 20
.sequential/conv2d_1/squeeze_batch_dims/BiasAdd�
6sequential/conv2d_1/squeeze_batch_dims/concat/values_1Const*
_output_shapes
:*
dtype0*!
valueB"          28
6sequential/conv2d_1/squeeze_batch_dims/concat/values_1�
2sequential/conv2d_1/squeeze_batch_dims/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������24
2sequential/conv2d_1/squeeze_batch_dims/concat/axis�
-sequential/conv2d_1/squeeze_batch_dims/concatConcatV2=sequential/conv2d_1/squeeze_batch_dims/strided_slice:output:0?sequential/conv2d_1/squeeze_batch_dims/concat/values_1:output:0;sequential/conv2d_1/squeeze_batch_dims/concat/axis:output:0*
N*
T0*
_output_shapes
:2/
-sequential/conv2d_1/squeeze_batch_dims/concat�
0sequential/conv2d_1/squeeze_batch_dims/Reshape_1Reshape7sequential/conv2d_1/squeeze_batch_dims/BiasAdd:output:06sequential/conv2d_1/squeeze_batch_dims/concat:output:0*
T0*3
_output_shapes!
:��������� 22
0sequential/conv2d_1/squeeze_batch_dims/Reshape_1�
sequential/conv2d_1/ReluRelu9sequential/conv2d_1/squeeze_batch_dims/Reshape_1:output:0*
T0*3
_output_shapes!
:��������� 2
sequential/conv2d_1/Relu�
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����    2
sequential/flatten/Const�
sequential/flatten/ReshapeReshape&sequential/conv2d_1/Relu:activations:0!sequential/flatten/Const:output:0*
T0*'
_output_shapes
:��������� 2
sequential/flatten/Reshape�
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

: d*
dtype02(
&sequential/dense/MatMul/ReadVariableOp�
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential/dense/MatMul�
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp�
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d2
sequential/dense/BiasAdd�
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������d2
sequential/dense/Relu�
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp�
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/dense_1/MatMul�
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp�
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
sequential/dense_1/BiasAdd�
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
sequential/dense_1/Softmax�
IdentityIdentity$sequential/dense_1/Softmax:softmax:0/^sequential/conv2d/Conv2D/Conv2D/ReadVariableOp<^sequential/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp1^sequential/conv2d_1/Conv2D/Conv2D/ReadVariableOp>^sequential/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:���������::::::::2`
.sequential/conv2d/Conv2D/Conv2D/ReadVariableOp.sequential/conv2d/Conv2D/Conv2D/ReadVariableOp2z
;sequential/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp;sequential/conv2d/squeeze_batch_dims/BiasAdd/ReadVariableOp2d
0sequential/conv2d_1/Conv2D/Conv2D/ReadVariableOp0sequential/conv2d_1/Conv2D/Conv2D/ReadVariableOp2~
=sequential/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp=sequential/conv2d_1/squeeze_batch_dims/BiasAdd/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:a ]
3
_output_shapes!
:���������
&
_user_specified_nameconv2d_input"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
Q
conv2d_inputA
serving_default_conv2d_input:0���������;
dense_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�3
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
*`&call_and_return_all_conditional_losses
a_default_save_signature
b__call__"�0
_tf_keras_sequential�0{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1, 1, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1, 1, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 1, 5]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1, 1, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1, 1, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"�	
_tf_keras_layer�	{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1, 1, 5]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1, 1, 1, 5]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 1, 5]}}
�	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1, 1, 1, 32]}}
�
	variables
trainable_variables
regularization_losses
	keras_api
*g&call_and_return_all_conditional_losses
h__call__"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
*i&call_and_return_all_conditional_losses
j__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
�

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
*k&call_and_return_all_conditional_losses
l__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 5, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100]}}
�
(iter

)beta_1

*beta_2
	+decay
,learning_ratemPmQmRmSmTmU"mV#mWvXvYvZv[v\v]"v^#v_"
	optimizer
X
0
1
2
3
4
5
"6
#7"
trackable_list_wrapper
X
0
1
2
3
4
5
"6
#7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
-layer_regularization_losses
.metrics
/layer_metrics
	variables
trainable_variables

0layers
1non_trainable_variables
	regularization_losses
b__call__
a_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
,
mserving_default"
signature_map
':% 2conv2d/kernel
: 2conv2d/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
2layer_regularization_losses
3metrics
4layer_metrics
	variables
trainable_variables

5layers
regularization_losses
6non_trainable_variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_1/kernel
: 2conv2d_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
7layer_regularization_losses
8metrics
9layer_metrics
	variables
trainable_variables

:layers
regularization_losses
;non_trainable_variables
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
<layer_regularization_losses
=metrics
>layer_metrics
	variables
trainable_variables

?layers
regularization_losses
@non_trainable_variables
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
: d2dense/kernel
:d2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Alayer_regularization_losses
Bmetrics
Clayer_metrics
	variables
trainable_variables

Dlayers
 regularization_losses
Enon_trainable_variables
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 :d2dense_1/kernel
:2dense_1/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Flayer_regularization_losses
Gmetrics
Hlayer_metrics
$	variables
%trainable_variables

Ilayers
&regularization_losses
Jnon_trainable_variables
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
'
K0"
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	Ltotal
	Mcount
N	variables
O	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
L0
M1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:,  2Adam/conv2d_1/kernel/m
 : 2Adam/conv2d_1/bias/m
#:! d2Adam/dense/kernel/m
:d2Adam/dense/bias/m
%:#d2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:,  2Adam/conv2d_1/kernel/v
 : 2Adam/conv2d_1/bias/v
#:! d2Adam/dense/kernel/v
:d2Adam/dense/bias/v
%:#d2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
�2�
G__inference_sequential_layer_call_and_return_conditional_losses_3103402
G__inference_sequential_layer_call_and_return_conditional_losses_3103324
G__inference_sequential_layer_call_and_return_conditional_losses_3103097
G__inference_sequential_layer_call_and_return_conditional_losses_3103122�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_3102926�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/
conv2d_input���������
�2�
,__inference_sequential_layer_call_fn_3103444
,__inference_sequential_layer_call_fn_3103423
,__inference_sequential_layer_call_fn_3103215
,__inference_sequential_layer_call_fn_3103169�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_conv2d_layer_call_and_return_conditional_losses_3103477�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv2d_layer_call_fn_3103486�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3103519�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_conv2d_1_layer_call_fn_3103528�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_flatten_layer_call_and_return_conditional_losses_3103534�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_flatten_layer_call_fn_3103539�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_layer_call_and_return_conditional_losses_3103550�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_layer_call_fn_3103559�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_dense_1_layer_call_and_return_conditional_losses_3103570�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_dense_1_layer_call_fn_3103579�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_3103246conv2d_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_3102926�"#A�>
7�4
2�/
conv2d_input���������
� "1�.
,
dense_1!�
dense_1����������
E__inference_conv2d_1_layer_call_and_return_conditional_losses_3103519t;�8
1�.
,�)
inputs��������� 
� "1�.
'�$
0��������� 
� �
*__inference_conv2d_1_layer_call_fn_3103528g;�8
1�.
,�)
inputs��������� 
� "$�!��������� �
C__inference_conv2d_layer_call_and_return_conditional_losses_3103477t;�8
1�.
,�)
inputs���������
� "1�.
'�$
0��������� 
� �
(__inference_conv2d_layer_call_fn_3103486g;�8
1�.
,�)
inputs���������
� "$�!��������� �
D__inference_dense_1_layer_call_and_return_conditional_losses_3103570\"#/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� |
)__inference_dense_1_layer_call_fn_3103579O"#/�,
%�"
 �
inputs���������d
� "�����������
B__inference_dense_layer_call_and_return_conditional_losses_3103550\/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������d
� z
'__inference_dense_layer_call_fn_3103559O/�,
%�"
 �
inputs��������� 
� "����������d�
D__inference_flatten_layer_call_and_return_conditional_losses_3103534d;�8
1�.
,�)
inputs��������� 
� "%�"
�
0��������� 
� �
)__inference_flatten_layer_call_fn_3103539W;�8
1�.
,�)
inputs��������� 
� "���������� �
G__inference_sequential_layer_call_and_return_conditional_losses_3103097|"#I�F
?�<
2�/
conv2d_input���������
p

 
� "%�"
�
0���������
� �
G__inference_sequential_layer_call_and_return_conditional_losses_3103122|"#I�F
?�<
2�/
conv2d_input���������
p 

 
� "%�"
�
0���������
� �
G__inference_sequential_layer_call_and_return_conditional_losses_3103324v"#C�@
9�6
,�)
inputs���������
p

 
� "%�"
�
0���������
� �
G__inference_sequential_layer_call_and_return_conditional_losses_3103402v"#C�@
9�6
,�)
inputs���������
p 

 
� "%�"
�
0���������
� �
,__inference_sequential_layer_call_fn_3103169o"#I�F
?�<
2�/
conv2d_input���������
p

 
� "�����������
,__inference_sequential_layer_call_fn_3103215o"#I�F
?�<
2�/
conv2d_input���������
p 

 
� "�����������
,__inference_sequential_layer_call_fn_3103423i"#C�@
9�6
,�)
inputs���������
p

 
� "�����������
,__inference_sequential_layer_call_fn_3103444i"#C�@
9�6
,�)
inputs���������
p 

 
� "�����������
%__inference_signature_wrapper_3103246�"#Q�N
� 
G�D
B
conv2d_input2�/
conv2d_input���������"1�.
,
dense_1!�
dense_1���������