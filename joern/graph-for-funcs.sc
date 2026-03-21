import scala.jdk.CollectionConverters._
import io.circe.syntax._
import io.circe.generic.semiauto._
import io.circe.{Encoder, Json}
import io.shiftleft.codepropertygraph.generated.nodes
import io.shiftleft.semanticcpg.language._
import overflowdb._
import java.io.PrintWriter

// Case classes for structured JSON output
final case class GraphForFuncsFunction(
    function: String, 
    file: String, 
    id: String, 
    AST: List[nodes.AstNode], 
    CFG: List[nodes.AstNode], 
    PDG: List[nodes.AstNode]
)

final case class GraphForFuncsResult(functions: List[GraphForFuncsFunction])

// JSON Encoders
implicit val encodeEdge: Encoder[overflowdb.Edge] = (edge: overflowdb.Edge) => Json.obj(
    ("id", Json.fromString(edge.toString)), 
    ("type", Json.fromString(edge.label())),
    ("in", Json.fromString(edge.inNode.id.toString)), 
    ("out", Json.fromString(edge.outNode.id.toString))
)

implicit val encodeNode: Encoder[nodes.AstNode] = (node: nodes.AstNode) => Json.obj(
    ("id", Json.fromString(node.id.toString)), 
    ("edges", Json.fromValues(
        (node.inE("AST", "CFG", "REACHING_DEF", "CDG").l ++ 
         node.outE("AST", "CFG", "REACHING_DEF", "CDG").l).map(_.asJson)
    )), 
    ("properties", Json.fromValues(node.propertiesMap.asScala.toList.map { case (key, value) => 
        Json.obj(("key", Json.fromString(key)), ("value", Json.fromString(value.toString))) 
    }))
)

implicit val encodeFuncFunction: Encoder[GraphForFuncsFunction] = deriveEncoder
implicit val encodeFuncResult: Encoder[GraphForFuncsResult] = deriveEncoder

// Logic to extract AST, CFG, and PDG components
val result = GraphForFuncsResult(cpg.method.map { method => 
    GraphForFuncsFunction(
        method.fullName, 
        method.filename, 
        method.id.toString, 
        method.astMinusRoot.l, 
        method.cfgNode.l, 
        (method.ast.isCfgNode.filter(n => n.outE("REACHING_DEF", "CDG").hasNext).l ++ 
         method.ast.isCfgNode.filter(n => n.inE("REACHING_DEF", "CDG").hasNext).l).distinct.cast[nodes.AstNode].l
    ) 
}.l).asJson

// Internal write to avoid Joern's Unit/() return type issue
val jsonString = result.noSpaces
val writer = new PrintWriter("last_graph_export.json")
try { writer.write(jsonString) } finally { writer.close() }