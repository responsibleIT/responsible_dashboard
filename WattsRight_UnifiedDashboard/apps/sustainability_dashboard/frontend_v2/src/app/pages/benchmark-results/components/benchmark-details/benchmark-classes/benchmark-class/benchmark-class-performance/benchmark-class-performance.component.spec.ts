import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkClassPerformanceComponent } from './benchmark-class-performance.component';

describe('BenchmarkClassPerformanceComponent', () => {
  let component: BenchmarkClassPerformanceComponent;
  let fixture: ComponentFixture<BenchmarkClassPerformanceComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BenchmarkClassPerformanceComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BenchmarkClassPerformanceComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
