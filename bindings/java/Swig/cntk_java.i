//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// cntk_java.i -- SWIG Interface file for Java
//

//JNI defines UNUSED macro as well, undefining it so it doesn't conflict with CNTK's
%{
#undef UNUSED
%}

%include "CNTKManagedCommon.i"

%pragma(java) jniclasscode=%{
  static {
    String libName = "Cntk.Core.JavaBinding-2.0rc2";
    try {
       System.loadLibrary(libName);
    } catch (UnsatisfiedLinkError e) {
       try {
           System.loadLibrary(libName+'d');
       } catch (UnsatisfiedLinkError e2) {
          System.err.println("Native code library failed to load. \n" + e2);
          System.exit(1);
       }
    }
  }
%}

// Java specific extention.
%typemap(javacode) CNTK::DeviceDescriptor %{
    public java.util.List<DeviceDescriptor> getAllDevices() {
        DeviceDescriptorVector devices = _AllDevices();
        java.util.ArrayList<DeviceDescriptor> ret = new java.util.ArrayList<DeviceDescriptor>((int)devices.size());
        for (int i = 0; i < devices.size(); ++i){
            ret.add(devices.get(i));
        }
        return ret;
    }

    public static DeviceDescriptor getCPUDevice() {
        return _CPUDevice();
    }

    public DeviceKind getDeviceType() {
        return _DeviceType();
    }

    public long getId() {
        return _Id();
    }

    public void setExcludedDevices(DeviceDescriptorVector ddv) {
        _SetExcludedDevices(ddv);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        DeviceDescriptor p = (DeviceDescriptor)o;
        if (p == null) return false;
        return CNTKLib.AreEqualDeviceDescriptor(this, p);
    }

    public boolean equals(DeviceDescriptor p) {
        if (p == null) return false;
        return CNTKLib.AreEqualDeviceDescriptor(this, p);
    }

    @Override
    public int hashCode() {
        return _DeviceType().hashCode();
    }
%}

%typemap(javacode) CNTK::Axis %{

    public String getName() {
        return _Name();
    }

    public boolean isOrdered() {
        return _IsOrdered();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        Axis p = (Axis)o;
        if (p == null) return false;
        return CNTKLib.AreEqualAxis(this, p);
    }

    public boolean equals(Axis p) {
        if (p == null) return false;
        return CNTKLib.AreEqualAxis(this, p);
    }

    @Override
    public int hashCode() {
        if (this.IsDynamicAxis()) {
            return _Name().hashCode();
        } else {
            return this.StaticAxisIndex();
        }
    }
%}


%typemap(javacode) CNTK::Function %{

    public String getName() {
        return _Name();
    }

    public String getUid() {
        return _Uid();
    }

    public Function getRootFunction() {
        return _RootFunction();
    }

    public static Function Load(byte[] modelBuffer, DeviceDescriptor computeDevice)
    {
        return Load(modelBuffer, (long)modelBuffer.length, computeDevice);
    }

    // TODO: look at C# implementation and make it look more like that
    private VariableVector argumentVector;
    private VariableVector outputVector;
    private VariableVector inputVector;
    private java.util.ArrayList<Variable> argumentList;
    private java.util.ArrayList<Variable> outputList;
    private java.util.ArrayList<Variable> inputList;

    private UnorderedMapVariableValuePtr outMap = new UnorderedMapVariableValuePtr();

    public java.util.List<Variable> getInputs() {
        if (inputVector == null) {
            inputVector = _Inputs();
            inputList = new java.util.ArrayList<Variable>((int)inputVector.size());
            for (int i = 0; i < inputVector.size(); ++i){
                inputList.add(inputVector.get(i));
            }
        }
        return inputList;
    }

    public Variable getOutput() {
        return _Output();
    }

    public java.util.List<Variable> getOutputs() {
        if (outputVector == null) {
            outputVector = _Outputs();
            outputList = new java.util.ArrayList<Variable>((int)outputVector.size());
            for (int i = 0; i < outputVector.size(); ++i){
                outputList.add(outputVector.get(i));
            }
        }
        return outputList;
    }

    public java.util.List<Variable> getArguments() {
        if (argumentVector == null) {
            argumentVector = _Arguments();
            argumentList = new java.util.ArrayList<Variable>((int)argumentVector.size());
            for (int i = 0; i < argumentVector.size(); ++i){
                argumentList.add(argumentVector.get(i));
            }
        }
        return argumentList;
    }

    public String getOpName() {
        return _OpName();
    }

    public Function clone() {
        return _Clone();
    }

    public FunctionPtrVector findAllWithName(String name) {
        return _FindAllWithName(name);
    }

    public boolean isComposite() {
        return _IsComposite();
    }

    public boolean isPrimitive() {
        return _IsPrimitive();
    }

    public boolean isBlock() {
        return _IsBlock();
    }

    public static Function Combine(java.util.ArrayList<Variable> outputVariable) {
        VariableVector varVect = new VariableVector();
        for (int i = 0; i < outputVariable.size(); ++i)
        {
            varVect.add(varVect.get(i));
        }
        return CNTKLib.Combine(varVect);
    }
%}

%typemap(javacode) CNTK::Variable %{

    public NDShape getShape() {
        return _Shape();
    }

    public String getName() {
        return _Name();
    }

    public VariableKind getVariableKind() {
        return _VariableKind();
    }

    public AxisVector getDynamicAxes() {
        return _DynamicAxes();
    }

    public boolean isSparse() {
        return _IsSparse();
    }

    public boolean isInput() {
        return _IsInput();
    }

    public boolean isOutput() {
        return _IsOutput();
    }

    public boolean isParameter() {
        return _IsParameter();
    }

    public boolean isConstant() {
        return _IsConstant();
    }

    public boolean isPlaceholder() {
        return _IsPlaceholder();
    }

    public Function getOwner() {
        return _Owner();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        Variable p = (Variable)o;
        if (p == null) return false;
        return CNTKLib.AreEqualVariable(this, p);
    }

    public boolean equals(Variable p) {
        if (p == null) return false;
        return CNTKLib.AreEqualVariable(this, p);
    }

    @Override
    public int hashCode() {
        return (int)GetHashValue();
    }
%}

%typemap(javacode) CNTK::NDShape %{

    public long getRank() {
        return _Rank();
    }

    public long getTotalSize() {
        return _TotalSize();
    }

    public boolean isUnknown() {
        return _IsUnknown();
    }

    public boolean hasInferredDimension() {
        return _HasInferredDimension();
    }

    public boolean hasFreeDimension() {
        return _HasFreeDimension();
    }

    public java.util.ArrayList<Long> getDimensions(){
        java.util.ArrayList<Long> ret = new java.util.ArrayList<Long>((int)_Rank());
        for (int i = 0; i < _Dimensions().size(); ++i ) {
            ret.add((Long)_Dimensions().get(i));
        }
        return ret;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null) return false;
        NDShape p = (NDShape)o;
        if (p == null) return false;
        return CNTKLib.AreEqualShape(this, p);
    }

    public boolean equals(NDShape p) {
        if (p == null) return false;
        return CNTKLib.AreEqualShape(this, p);
    }

    @Override
    public int hashCode() {
        return _Dimensions().hashCode();
    }
%}

%typemap(javacode) CNTK::NDMask %{

    public long getMaskedCount() {
        return _MaskedCount();
    }

    public DeviceDescriptor getDevice() {
        return _Device();
    }

    public NDShape getShape() {
        return _Shape();
    }

    public void invalidateSection(SizeTVector sectionOffset, NDShape sectionShape) {
        _InvalidateSection(sectionOffset, sectionShape);
    }

    public void markSequenceBegin(SizeTVector offset) {
        _MarkSequenceBegin(offset);
    }
%}

%typemap(javacode) CNTK::Value %{
    public DeviceDescriptor getDevice() {
        return _Device();
    }

    public NDShape getShape() {
        return _Shape();
    }

    public boolean isSparse() {
        return _IsSparse();
    }

    public boolean isReadOnly() {
        return _IsReadOnly();
    }

    public long getMaskedCount() {
        return _MaskedCount();
    }
%}

%typemap(javacode) CNTK::NDArrayView %{
    public DeviceDescriptor getDevice() {
        return _Device();
    }

    public NDShape getShape() {
        return _Shape();
    }

    public boolean isSparse() {
        return _IsSparse();
    }

    public boolean isReadOnly() {
        return _IsReadOnly();
    }

    public NDArrayView getSliceView(SizeTVector startOffset, SizeTVector extent) {
        return _SliceView(startOffset, extent);
    }
%}

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"
