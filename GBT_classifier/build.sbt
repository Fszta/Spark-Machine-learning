name := "untitled1"

version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark"  %%  "spark-core"    % "2.2.0"   % "provided",
  "org.apache.spark"  %%  "spark-sql"     % "2.2.0",
  "org.apache.spark"  %%  "spark-ml"      % "2.2.0",
  "com.typesafe.akka" %% "akka-http"   % "10.1.10",
  "com.typesafe.akka" %% "akka-actor" % "2.5.23",
  "com.typesafe.akka" %% "akka-stream" % "2.5.23"
)