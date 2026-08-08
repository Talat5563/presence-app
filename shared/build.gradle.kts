plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

kotlin {
    jvmToolchain(17)

    jvm()

    sourceSets {
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
        jvmMain.dependencies {
            implementation(libs.onnxruntime)
        }
    }
}

// Two suites (see Global Constraints). The default suite runs pure-logic tests only; anything
// needing a real model file is tagged "model" and runs under `integrationTest`, which fails
// loudly when the model is absent rather than skipping silently.
//
// The exclusion below must skip the `integrationTest` task itself (registered below): applying
// `excludeTags("model")` via `withType<Test>().configureEach` also configures `integrationTest`
// (it is a `Test` task too), and JUnit Platform treats a tag that is both included and excluded
// as excluded — silently running zero tests under `integrationTest` forever. That is exactly the
// silent-skip behavior forbidden by the global test-suite contract.
tasks.withType<Test>().configureEach {
    if (name != "integrationTest") {
        useJUnitPlatform {
            excludeTags("model")
        }
    }
}

tasks.register<Test>("integrationTest") {
    description = "Runs tests that require the ONNX model files in models/."
    group = "verification"
    val jvmTest = tasks.named<Test>("jvmTest").get()
    testClassesDirs = jvmTest.testClassesDirs
    classpath = jvmTest.classpath
    useJUnitPlatform {
        includeTags("model")
    }
}
