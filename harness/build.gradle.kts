plugins {
    alias(libs.plugins.kotlinJvm)
    application
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(project(":shared"))
    testImplementation(kotlin("test"))
}

application {
    mainClass.set("app.presence.harness.MainKt")
}

// Same two-suite split as :shared — see Global Constraints.
tasks.test {
    useJUnitPlatform {
        excludeTags("model")
    }
}

tasks.register<Test>("integrationTest") {
    description = "Runs tests that require the ONNX model files in models/."
    group = "verification"
    testClassesDirs = sourceSets.test.get().output.classesDirs
    classpath = sourceSets.test.get().runtimeClasspath
    useJUnitPlatform {
        includeTags("model")
    }
}
