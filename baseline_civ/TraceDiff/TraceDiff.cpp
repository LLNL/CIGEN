#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Function.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/DebugInfoMetadata.h"

#include <iostream>
#include <sstream>
using namespace llvm;
using namespace llvm::legacy;

#define DEBUG_PRINT 0

namespace {
  void injectFPProfileCall(Instruction& I, BasicBlock& BB, IRBuilder<>& builder, Module* module) {
    builder.SetInsertPoint(&I);
    // Declare C standard library printf 
    Type *intType = Type::getInt32Ty(module->getContext());
    std::vector<Type *> printfArgsTypes({Type::getInt8PtrTy(module->getContext())});
    FunctionType *printfType = FunctionType::get(intType, printfArgsTypes, true);
    FunctionCallee printfFunc = module->getOrInsertFunction("printf", printfType);
    std::string printStr = "ins ";
    printStr += I.getOpcodeName();
    printStr += "\n";
    Value *str = builder.CreateGlobalStringPtr(printStr.c_str(), printStr.c_str());
    std::vector<Value *> argsV({str});
    builder.CreateCall(printfFunc, argsV, "calltmp");
  }

  bool traceDiffFunction(Function &F) {
    Module* module = F.getParent();
    IRBuilder<> builder(module->getContext());

    for (BasicBlock &BB : F) {
      #if DEBUG_PRINT
      errs() << "Basic block (name=" << BB.getName() << ") has "
                << BB.size() << " instructions.\n";
      #endif
      int insIndex = 0;
      for (Instruction &I : BB) {
        #if DEBUG_PRINT
        errs() << "Instruction " << insIndex << ":" << I << "\n";
        #endif
        if (I.getOpcode() == Instruction::PHI)
          continue;
        bool hasFloat = false;
        for (unsigned int opI = 0; opI < I.getNumOperands(); opI++) {
          Value* op = I.getOperand(opI);
          if (op->getType()->isFPOrFPVectorTy()) {
            hasFloat = true;
          }
          if (LoadInst *LMemI = dyn_cast<LoadInst>(&I)) {
            Value* PtrValue = LMemI->getPointerOperand();
            Type* PointerElementType = LMemI->getType();
            if (PointerElementType->isFPOrFPVectorTy()) {
              hasFloat = true;
            }
          }
          #if DEBUG_PRINT
          errs() << "op " << opI << ":" << *op << "\n";
          #endif
        }
        if (hasFloat) {
          #if DEBUG_PRINT
          errs() << "has float\n";
          #endif
          injectFPProfileCall(I, BB, builder, module);
        }
        insIndex++;
      }
    }
    return false;    
  }

  struct TraceDiffPass : public FunctionPass {
    static char ID;
    TraceDiffPass() : FunctionPass(ID) {}

    virtual bool runOnFunction(Function &F) {
		  return traceDiffFunction(F);
    }

    virtual bool doFinalization(Module& M) {
      return false;
    }
  };

  struct TraceDiff : public PassInfoMixin<TraceDiff> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
      if (!traceDiffFunction(F))
        return PreservedAnalyses::all();
      return PreservedAnalyses::none();
    }
  
    static bool isRequired() {
      return true;
    }
  };
}

PassPluginLibraryInfo getTraceDiffPassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    PB.registerPipelineEarlySimplificationEPCallback (
        [&](ModulePassManager &MPM, OptimizationLevel opt) {
		  errs() << "opt:" << opt.getSpeedupLevel() << "\n";
          MPM.addPass(createModuleToFunctionPassAdaptor(TraceDiff()));
          return true;
        });
  };

  return {LLVM_PLUGIN_API_VERSION, "name", "0.0.1", callback};
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getTraceDiffPassPluginInfo();
}