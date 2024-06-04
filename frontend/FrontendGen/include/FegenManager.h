#ifndef FEGEN_MANAGER_H
#define FEGEN_MANAGER_H

#include <string>
#include <vector>
#include <map>
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"
#include "FegenParser.h"

namespace fegen {

class FegenType;

// user defined function
class FegenFunction {
private:
    // cpp function name
    std::string name;
    // input types
    std::vector<FegenType*> inputTypeList;
    // return type
    FegenType* returnType;
    explicit FegenFunction(llvm::StringRef name, std::vector<FegenType*>&& inputTypeList, FegenType* returnType);
public:
    static FegenFunction* get(llvm::StringRef name, std::vector<FegenType*> inputTypeList, FegenType* returnType = nullptr);
    ~FegenFunction() = default;
};

class FegenValue;

// user defined operation
class FegenOperation {
private:
    std::string dialectName;
    std::string operationName;
    // arguments of operation
    std::vector<FegenValue*> arguments;
    // results of operation
    std::vector<FegenValue*> results;
    explicit FegenOperation(llvm::StringRef dialectName, llvm::StringRef operationName, std::vector<FegenValue*>&& arguments, std::vector<FegenValue*>&& results);
public:
    static FegenOperation* get(llvm::StringRef dialectName, llvm::StringRef operationName, std::vector<FegenValue*> arguments, std::vector<FegenValue*> results);
    ~FegenOperation()=default;
};

class FegenType {
public:
enum class TypeKind{
    ATTRIBUTE_VALUE,
    OPERAND_VALUE,
    CPP_VALUE
};
private:
    TypeKind kind;
    std::string dialectName;
    std::string typeName;
    std::string assemblyFormat;
    std::vector<FegenValue*> parameters;
    // context of type in parser tree
    FegenParser::TypeDefinationDeclContext* ctx;
    explicit FegenType(TypeKind kind, llvm::StringRef dialectName, llvm::StringRef typeName, llvm::StringRef assemblyFormat, std::vector<FegenValue*>&& parameters, FegenParser::TypeDefinationDeclContext* ctx);
public:
    // convert from fegen type to cpp type
    std::string convertToCppType();
    static FegenType* get(TypeKind kind, llvm::StringRef dialectName, llvm::StringRef typeName, llvm::StringRef assemblyFormat, std::vector<FegenValue*> parameters, FegenParser::TypeDefinationDeclContext* ctx);
    ~FegenType()=default;
};

class FegenValue {

private:
    bool isList;
    FegenType* type;
    std::string name;
    antlr4::ParserRuleContext* ctx;
    explicit FegenValue(bool isList, FegenType* type, llvm::StringRef name, antlr4::ParserRuleContext* ctx);
public:
    static FegenValue* get(bool isList, FegenType* type, llvm::StringRef name, antlr4::ParserRuleContext* ctx);
    ~FegenValue()=default;
};

class FegenNode;

class FegenRule {
private:
    std::string content;
    // from which node
    FegenNode* src;
    std::vector<FegenValue*> inputs;
    std::vector<FegenValue*> returns;
    // context in parser tree
    FegenParser::ActionAltContext* ctx;
    explicit FegenRule(std::string content, FegenNode* src, std::vector<FegenValue*>&& inputs, std::vector<FegenValue*>&& returns, FegenParser::ActionAltContext* ctx);
public:
    static FegenRule* get(std::string content, FegenNode* src, std::vector<FegenValue*> inputs, std::vector<FegenValue*> returns, FegenParser::ActionAltContext* ctx);
};

class FegenNode {
private:
    std::vector<FegenRule*> rules;
    FegenParser::RuleSpecContext* ctx;
    explicit FegenNode(std::vector<FegenRule*>&& rules, FegenParser::RuleSpecContext* ctx);
public:
    static FegenNode* get(std::vector<FegenRule*> rules, FegenParser::RuleSpecContext* ctx);
    static FegenNode* get(FegenParser::RuleSpecContext* ctx);
    void addFegenRule(FegenRule* rule);
    // release rules first
    ~FegenNode();
};

class FegenVisitor;

class FegenManager{
friend class FegenVisitor;
private:
    std::vector<std::string> headFiles;
    std::map<llvm::StringRef, FegenNode*> nodeMap;
    llvm::StringMap<FegenType*> typeMap;
    llvm::StringMap<FegenOperation*> operationMap;
    llvm::StringMap<FegenFunction*> functionMap;
public:
    // release nodes, type, operation, function
    ~FegenManager();

};

} // namespace fegen

#endif