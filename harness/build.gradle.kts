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

// Relative paths in --args resolve against the repository root, as the README's commands assume.
tasks.named<JavaExec>("run") {
    workingDir = rootProject.projectDir
}

tasks.register<JavaExec>("importChildes") {
    description = "Converts CHILDES transcripts and media into a harness corpus."
    group = "application"
    classpath = sourceSets.main.get().runtimeClasspath
    mainClass.set("app.presence.harness.ChildesImport")
    // Holds a few decoded hour-long sessions at once.
    maxHeapSize = "3g"
    javaLauncher.set(javaToolchains.launcherFor { languageVersion.set(JavaLanguageVersion.of(17)) })
    workingDir = rootProject.projectDir
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
